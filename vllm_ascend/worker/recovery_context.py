from queue import Queue
from vllm_ascend.worker.common import FaultToleranceLevel

class RecoveryContext:
    def __init__(self,model,level:FaultToleranceLevel,exception : 'Exception',rank: int,model_or_path:'str',
                 fault_queue:'Queue'):
        self.model = model
        self.level = level
        self.exception = exception
        self.rank = rank
        self.model_or_path = model_or_path
        self.fault_queue = fault_queue

