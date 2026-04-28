from typing import Optional
import logging

import torch.distributed as dist

# 全局并行组引用（与 vLLM 内部保持一致）
from vllm.distributed.parallel_state import (
    _PP, _DP, _TP, _PCP, _DCP, _EP,
    get_world_group,
    init_model_parallel_group,
    get_current_vllm_config
)

logger = logging.getLogger(__name__)

def destroy_parallel_group(group_name: str) -> bool:
    """
    安全销毁指定的并行通信组。
    
    Args:
        group_name: 要销毁的并行组名称，支持: "PP", "DP", "TP", "PCP", "DCP", "EP"
    
    Returns:
        True: 销毁成功
        False: 销毁失败（组不存在或已销毁）
    
    Raises:
        ValueError: 无效的组名称
        RuntimeError: 销毁过程中发生错误
    """
    # 获取全局组引用
    global_group_ref = globals().get(f"_{group_name}")
    if global_group_ref is None:
        logger.warning(f"{group_name} group is not initialized, nothing to destroy")
        return False
    
    try: 
        # 1. 调用组的 destroy 方法
        logger.info(f"Destroying {group_name} group...")
        global_group_ref.destroy()
        
        # 2. 置空全局变量
        globals()[f"_{group_name}"] = None
        
        logger.info(f"Successfully destroyed {group_name} group")
        return True
        
    except Exception as e:
        logger.error(f"Failed to destroy {group_name} group: {e}")
        raise RuntimeError(f"Failed to destroy {group_name} group") from e


def rebuild_parallel_group(
    group_name: str,
    parallel_config: Optional["ParallelConfig"] = None
) -> bool:
    """
    重建指定的并行通信组。
    
    Args:
        group_name: 要重建的并行组名称，支持: "PP", "DP", "TP", "PCP", "DCP", "EP"
        parallel_config: 并行配置对象。如果为 None，将从当前配置获取
    
    Returns:
        True: 重建成功
        False: 重建失败
    
    Raises:
        ValueError: 无效的组名称
        RuntimeError: 重建过程中发生错误
    """
    # 获取配置
    if parallel_config is None:
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        parallel_config = config.parallel_config
    
    # 获取当前 rank 和 world size
    world_group = get_world_group()
    rank = world_group.rank if world_group else dist.get_rank()
    world_size = world_group.world_size if world_group else dist.get_world_size()
    local_rank = world_group.local_rank if world_group else 0
    
    # 获取 backend
    backend = dist.get_backend(world_group.device_group) if world_group else "hccl"
    
    try:
        # 1. 确保旧组已销毁
        destroy_parallel_group(group_name)
        
        # 2. 构建 rank 列表（与 vLLM 初始化逻辑保持一致）
        logger.info(f"Preparing to rebuild {group_name} group...")
        
        # 获取并行配置参数
        dp_size = parallel_config.data_parallel_size
        pp_size = parallel_config.pipeline_model_parallel_size
        pcp_size = parallel_config.prefill_context_model_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        dcp_size = parallel_config.decode_context_model_parallel_size or 1
        
        # 构建 all_ranks 张量（与 initialize_model_parallel 保持一致）
        all_ranks = torch.arange(world_size).reshape(
            -1,  # ExternalDP
            dp_size,
            pp_size,
            pcp_size,
            tp_size,
        )
        
        # 根据组类型计算 group_ranks
        group_ranks = _compute_group_ranks(
            group_name, all_ranks, 
            dp_size, pp_size, pcp_size, tp_size, dcp_size
        )
        
        # 3. 初始化组
        logger.info(f"Initializing {group_name} group with ranks: {group_ranks}")
        
        # 特殊处理某些组（如 TP 需要 message queue broadcaster）
        use_mq_broadcaster = group_name in ["TP", "DCP"]
        
        new_group = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=local_rank,
            backend=backend,
            use_message_queue_broadcaster=use_mq_broadcaster,
            group_name=group_name.lower()
        )
        
        # 4. 设置全局变量
        globals()[f"_{group_name}"] = new_group
        
        logger.info(f"Successfully rebuilt {group_name} group: "
                    f"rank={new_group.rank_in_group}, size={new_group.world_size}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to rebuild {group_name} group: {e}")
        # 清理可能的部分初始化
        destroy_parallel_group(group_name)
        raise RuntimeError(f"Failed to rebuild {group_name} group") from e

def _compute_group_ranks(
    group_name: str,
    all_ranks: torch.Tensor,
    dp_size: int,
    pp_size: int,
    pcp_size: int,
    tp_size: int,
    dcp_size: int
) -> list[list[int]]:
    """
    计算指定并行组的 rank 列表（与 vLLM 内部逻辑保持一致）。
    """
    if group_name == "TP":
        # TP: 按张量并行维度分组
        group_ranks = all_ranks.view(-1, tp_size).unbind(0)
        
    elif group_name == "PP":
        # PP: 按流水线并行维度分组
        group_ranks = (all_ranks.transpose(2, 4)
                    .reshape(-1, pp_size)
                    .unbind(0))
        
    elif group_name == "DP":
        # DP: 按数据并行维度分组
        group_ranks = (all_ranks.transpose(1, 4)
                    .reshape(-1, dp_size)
                    .unbind(0))
        
    elif group_name == "PCP":
        # PCP: 按预填充上下文并行维度分组
        group_ranks = (all_ranks.transpose(3, 4)
                    .reshape(-1, pcp_size)
                    .unbind(0))
        
    elif group_name == "DCP":
        # DCP: 按解码上下文并行维度分组
        group_ranks = all_ranks.reshape(-1, dcp_size).unbind(0)
        
    elif group_name == "EP":
        # EP: 按专家并行维度分组
        ep_size = dp_size * pcp_size * tp_size
        group_ranks = (all_ranks.transpose(1, 2)
                    .reshape(-1, ep_size)
                    .unbind(0))
    
    return [x.tolist() for x in group_ranks]