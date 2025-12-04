from queue import Queue
from vllm_ascend.worker.common import FaultToleranceLevel

class RecoveryContext:
    def __init__(self,model,level:FaultToleranceLevel,exception : 'Exception',rank: int,world_size:int,
                 fault_queue:'Queue'):
        self.model = model
        self.level = level
        self.exception = exception
        self.rank = rank
        self.world_size = world_size
        self.fault_queue = fault_queue

