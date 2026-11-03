"""Exercise real spawned processes without downloading a model or importing CUDA."""
import multiprocessing as mp
import time
import unittest
from unittest.mock import patch

from model_worker import ModelRunner


def echo_worker(connection, model_path):
    connection.send(('ready', ''))
    while True:
        message = connection.recv()
        connection.send(('answer', str(message)))


class LifecycleTests(unittest.TestCase):
    def wait_for(self, runner, expected):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = runner.poll()
            if result and result[0] == expected:
                return result
            time.sleep(0.01)
        self.fail(f'Timed out waiting for {expected}')

    def test_lazy_start_reuse_and_idle_termination(self):
        runner = ModelRunner('unused', idle_seconds=120)
        self.assertIsNone(runner.process)
        try:
            with patch('model_worker.worker', echo_worker):
                runner.submit([{'role': 'user', 'content': 'hello'}])
            self.wait_for(runner, 'answer')
            self.assertTrue(runner.ready)
            self.assertFalse(runner.busy)
            pid = runner.process.pid
            runner.submit([{'role': 'user', 'content': 'again'}])
            self.wait_for(runner, 'answer')
            self.assertEqual(runner.process.pid, pid)
            runner.last_activity = time.monotonic() - 121
            self.assertEqual(runner.poll(), ('unloaded', ''))
            self.assertIsNone(runner.process)
            self.assertNotIn(pid, [p.pid for p in mp.active_children()])
            self.assertFalse(runner.ready)
        finally:
            runner.stop()

    def test_stop_while_loading_discards_private_queued_input(self):
        runner = ModelRunner('unused')
        try:
            with patch('model_worker.worker', echo_worker):
                runner.submit([{'role': 'user', 'content': 'temporary'}])
            self.assertTrue(runner.busy)
            runner.stop()
            self.assertIsNone(runner.queued)
            self.assertIsNone(runner.connection)
            self.assertIsNone(runner.process)
            self.assertFalse(runner.busy)
        finally:
            runner.stop()


if __name__ == '__main__':
    unittest.main()
