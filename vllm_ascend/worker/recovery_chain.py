import torch
import yaml
import torch_npu

from abc import ABC, abstractmethod
from vllm.logger import logger
from vllm_ascend.worker.common import RecoveryStatus,FaultStatus
from vllm_ascend.worker.recovery_context import RecoveryContext

force_stop_error = ["force stop"]
network_error = [
    "suspect remote error",
    "hccl op retry failed"
]


class RecoveryHandler(ABC):

    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'RecoveryHandler') -> 'RecoveryHandler':
        """Set next handler"""
        self.next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, ctx:RecoveryContext) -> bool:
        pass

    @abstractmethod
    def recover(self, ctx:RecoveryContext) -> torch.Tensor:
        """Specific recovery function"""
        pass

    def handle(self, ctx:RecoveryContext) -> torch.Tensor:
        """ Entry point for RecoveryHandler """
        if self.can_handle(ctx):
            return self.recover(ctx)
        elif self.next_handler:
            return self.next_handler.handle(ctx)
        else:
            logger.warning(f"No handler can process the exception:{ctx.exception}")
            raise ctx.exception


class ForceStopHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in force_stop_error:
            if error in error_str:
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """Force stop needs no extra recovery"""
        return RecoveryStatus.SUCCESS

class NetworkHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in network_error:
            if error in error_str:
                ctx.fault_queue.put_nowait(FaultStatus.NETWORK_ERR)
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """Network needs no extra recovery"""
        return RecoveryStatus.SUCCESS