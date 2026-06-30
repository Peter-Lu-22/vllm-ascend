# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This patch enhances MessageQueue.reset() and SpinCondition to clear ZMQ
# socket buffers during fault recovery. The upstream reset() only clears
# ShmRingBuffer metadata and current_idx, but leaves stale messages in
# ZMQ PUB/SUB and XPUB/SUB sockets, which can cause readers to process
# stale data after recovery.
#
# Controlled by VLLM_ASCEND_ENABLE_RECOVERY environment variable.

import time

import zmq
from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    SpinCondition,
)
from vllm.logger import logger


def _spin_condition_reset(self):
    """Reset SpinCondition ZMQ socket buffers for fault recovery.

    Reader side: drain stale pings from local_notify_socket (SUB) and
    cancel signals from read_cancel_socket (PAIR), then update last_read
    to avoid immediately entering idle mode.

    Writer side: close and rebind the local_notify_socket (PUB) to clear
    any stale messages held in the send buffer (SNDHWM=1).

    This is safe to call when no concurrent wait()/notify() is in flight
    (guaranteed by the recovery pause protocol).
    """
    if self.is_reader:
        # Update last_read so we stay in busy-loop mode after reset.
        self.last_read = time.monotonic()

        # Drain stale pings from the SUB socket.
        if self.local_notify_socket is not None:
            drained = 0
            while True:
                try:
                    self.local_notify_socket.recv(flags=zmq.NOBLOCK)
                    drained += 1
                except zmq.Again:
                    break
            if drained > 0:
                logger.debug("SpinCondition: drained %d stale pings", drained)

        # Drain stale cancel signals from the PAIR socket.
        if self.read_cancel_socket is not None:
            while True:
                try:
                    self.read_cancel_socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
    else:
        # Writer side: close + rebind PUB to clear send buffer.
        if self.local_notify_socket is not None:
            old_addr = self.local_notify_socket.getsockopt(
                zmq.LAST_ENDPOINT
            ).decode()
            self.local_notify_socket.setsockopt(zmq.LINGER, 0)
            self.local_notify_socket.close()

            # Reuse the same context by creating a fresh socket.
            # The PUB socket doesn't store a reference to the context,
            # so we use Context.instance() which returns the thread-local
            # singleton used by vLLM's ZMQ code.
            ctx = zmq.Context.instance()
            self.local_notify_socket = ctx.socket(zmq.PUB)
            self.local_notify_socket.setsockopt(zmq.SNDHWM, 1)
            self.local_notify_socket.bind(old_addr)
            logger.debug("SpinCondition: writer PUB rebound to %s", old_addr)


def _message_queue_reset(self):
    """Enhanced reset that also clears ZMQ socket buffers.

    Builds on the upstream reset() logic (clear ShmRingBuffer metadata,
    reset current_idx, update last_read) and adds:
    - Writer side: close + rebind local_socket (XPUB) to clear send buffer
    - Reader side: close + reconnect local_socket (SUB) to clear recv buffer
    - Both sides: reset SpinCondition to clear notify ping buffers
    """
    self.shutting_down = False

    if self._is_writer or self._is_local_reader:
        assert self.buffer is not None, "No buffer to reset"
        self.buffer.reset()
        self.current_idx = 0

        # Reset SpinCondition (drain/rebind notify sockets).
        if self._spin_condition is not None:
            _spin_condition_reset(self._spin_condition)

        # Reset local_socket (XPUB for writer, SUB for local reader).
        if self.local_socket is not None:
            if self._is_writer:
                # Writer (XPUB): close + rebind to clear send buffer.
                old_addr = self.local_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                self.local_socket.setsockopt(zmq.LINGER, 0)
                self.local_socket.close()
                ctx = zmq.Context.instance()
                self.local_socket = ctx.socket(zmq.XPUB)
                self.local_socket.setsockopt(zmq.XPUB_VERBOSE, True)
                self.local_socket.bind(old_addr)
                logger.debug("MessageQueue: writer XPUB rebound to %s", old_addr)
            elif self._is_local_reader:
                # Reader (SUB): close + reconnect to clear recv buffer.
                # Must re-subscribe since subscription is per-socket.
                socket_addr = self.handle.local_subscribe_addr
                self.local_socket.setsockopt(zmq.LINGER, 0)
                self.local_socket.close()
                ctx = zmq.Context.instance()
                self.local_socket = ctx.socket(zmq.SUB)
                self.local_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self.local_socket.connect(socket_addr)
                logger.debug("MessageQueue: reader SUB reconnected to %s", socket_addr)

    # Update spin_condition last_read for local readers.
    if self._is_local_reader and self._spin_condition is not None:
        self._spin_condition.last_read = time.monotonic()


# Apply patches.
SpinCondition.reset = _spin_condition_reset
MessageQueue.reset = _message_queue_reset
