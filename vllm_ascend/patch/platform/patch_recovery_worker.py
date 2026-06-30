# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This patch modifies WorkerProc to handle recovery from exceptions:
# 1. enqueue_output: detect NPU exceptions during get_output()
# 2. worker_busy_loop: pause when in_recovery (stop dequeuing rpc_broadcast_mq)
# 3. handle_output + async_output_busy_loop: track in-flight async outputs
#    so that reset_worker_mqs can wait for all enqueues to complete.
#
# Controlled by VLLM_ASCEND_ENABLE_RECOVERY environment variable.
import threading
import time
from functools import partial
from typing import Any

import cloudpickle
import msgspec.msgpack
import zmq

from vllm.logger import logger
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_ascend.recovery.types import ExceptionInfo


# ---------------------------------------------------------------------------
# 1. enqueue_output: detect NPU exceptions during AsyncModelRunnerOutput
# ---------------------------------------------------------------------------
def enqueue_output(self, output: Any):
    """Prepares output from the worker and enqueues it to the
    worker_response_mq. If the output is an Exception, it is
    converted to a FAILURE response.
    """
    if isinstance(output, AsyncModelRunnerOutput):
        try:
            output = output.get_output()
        except Exception as e:
            logger.error("[WorkerProc] Enqueue_output detected exception, send to WorkerMonitor")
            self.worker.worker.exception_occur = True
            if not self.worker.worker.in_recovery:
                self.worker.worker.in_recovery = True
                exception_info = ExceptionInfo(
                    exception_type=type(e).__name__,
                    exception_msg=str(e),
                )
                exception_encode = msgspec.msgpack.encode(exception_info)
                self.worker.worker_input_socket.send(exception_encode)
            output = e
    if isinstance(output, Exception):
        result = (WorkerProc.ResponseStatus.FAILURE, str(output))
    else:
        result = (WorkerProc.ResponseStatus.SUCCESS, output)
    if (response_mq := self.worker_response_mq) is not None:
        response_mq.enqueue(result)


# ---------------------------------------------------------------------------
# 2. handle_output: track in-flight async outputs with a counter + event
# ---------------------------------------------------------------------------
def handle_output(self, output: Any):
    """Handles output from the worker. If async scheduling is enabled,
    it is passed to the async_output_busy_loop thread. Otherwise, it is
    enqueued directly to the worker_response_mq.

    When async scheduling is enabled, increments _async_in_flight before
    putting to the queue so that recovery code can wait for all pending
    enqueues to complete before resetting worker_response_mq.
    """
    if self.use_async_scheduling:
        with self._async_in_flight_lock:
            self._async_in_flight += 1
        self.async_output_queue.put(output)
    else:
        self.enqueue_output(output)


# ---------------------------------------------------------------------------
# 3. async_output_busy_loop: decrement in-flight counter after enqueue
# ---------------------------------------------------------------------------
def async_output_busy_loop(self):
    """Entrypoint for the thread which handles outputs asynchronously.

    After each enqueue_output, decrements _async_in_flight and signals
    _async_drained_event when it reaches zero, allowing recovery code to
    know that no more enqueues are in flight.
    """
    from vllm.platforms import current_platform

    if hasattr(self.worker, "device"):
        current_platform.set_device(self.worker.device)

    while True:
        output = self.async_output_queue.get()
        self.enqueue_output(output)
        with self._async_in_flight_lock:
            self._async_in_flight -= 1
            if self._async_in_flight == 0:
                self._async_drained_event.set()


# ---------------------------------------------------------------------------
# 4. worker_busy_loop: pause when worker is in_recovery
# ---------------------------------------------------------------------------
def worker_busy_loop(self):
    """Main busy loop for Multiprocessing Workers.

    When the worker enters recovery (in_recovery=True), this loop pauses
    dequeuing from rpc_broadcast_mq and sets _busy_loop_paused so that
    recovery actions can safely drain/reset the message queues.
    """
    assert self.rpc_broadcast_mq is not None
    while True:
        # Check if worker is in recovery; if so, pause.
        if getattr(self.worker, "in_recovery", False):
            self._busy_loop_paused = True
            logger.info("[WorkerProc] worker_busy_loop paused for recovery")
            while getattr(self.worker, "in_recovery", False):
                time.sleep(0.1)
            self._busy_loop_paused = False
            logger.info("[WorkerProc] worker_busy_loop resumed after recovery")
            continue

        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
            indefinite=True
        )
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)

            output = func(*args, **kwargs)
        except Exception as e:
            import traceback
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception("WorkerProc hit an exception.")
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)
            continue

        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)


# ---------------------------------------------------------------------------
# 5. wait_for_async_drain: helper used by recovery actions
# ---------------------------------------------------------------------------
def wait_for_async_drain(self, timeout: float = 30.0) -> bool:
    """Wait until all in-flight async outputs have been enqueued.

    Returns True if drained within timeout, False otherwise.
    Called by the worker_pause_async_output recovery action.
    """
    if not self.use_async_scheduling:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with self._async_in_flight_lock:
            in_flight = self._async_in_flight
            queue_empty = self.async_output_queue.empty()
        if in_flight == 0 and queue_empty:
            return True
        time.sleep(0.1)
    return False


def wait_for_busy_loop_pause(self, timeout: float = 10.0) -> bool:
    """Wait until worker_busy_loop has paused.

    Returns True if paused within timeout, False otherwise.
    Called by the worker_pause_async_output recovery action.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(self, "_busy_loop_paused", False):
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

# Inject in-flight tracking attributes into __init__.
# We wrap the original __init__ to add our attributes after it runs.
_orig_init = WorkerProc.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._async_in_flight = 0
    self._async_in_flight_lock = threading.Lock()
    self._async_drained_event = threading.Event()
    self._async_drained_event.set()  # initially drained (nothing in flight)
    self._busy_loop_paused = False


WorkerProc.__init__ = _patched_init
WorkerProc.enqueue_output = enqueue_output
WorkerProc.handle_output = handle_output
WorkerProc.async_output_busy_loop = async_output_busy_loop
WorkerProc.worker_busy_loop = worker_busy_loop
WorkerProc.wait_for_async_drain = wait_for_async_drain
WorkerProc.wait_for_busy_loop_pause = wait_for_busy_loop_pause