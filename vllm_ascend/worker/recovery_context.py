from queue import Queue
from vllm_ascend.worker.common import FaultToleranceLevel

class RecoveryContext:
    def __init__(self,exception : 'Exception',fault_queue:'Queue'):
        self.exception = exception
        self.fault_queue = fault_queue

