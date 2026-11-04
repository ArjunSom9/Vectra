# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic batching groups concurrent requests and isolates failures."""

import asyncio
import time

import pytest

from serving import predict


class _Recorder:
    def __init__(self, fail_on=None, delay_s=0.0):
        self.batches = []
        self._fail_on = fail_on
        self._delay_s = delay_s

    def __call__(self, items):
        self.batches.append(list(items))
        time.sleep(self._delay_s)
        if self._fail_on is not None and self._fail_on in items:
            raise ValueError(f'bad item {self._fail_on}')
        return [item * 10 for item in items]


async def _submit_all(batcher, items):
    return await asyncio.gather(
        *(batcher.submit(i) for i in items), return_exceptions=True
    )


def test_concurrent_requests_share_batches_up_to_the_limit():
    recorder = _Recorder()

    async def scenario():
        batcher = predict.Batcher(recorder, max_batch_size=4, window_s=0.05)
        batcher.start()
        try:
            return await _submit_all(batcher, range(10))
        finally:
            await batcher.close()

    results = asyncio.run(scenario())
    assert [r.value for r in results] == [i * 10 for i in range(10)]
    assert [len(b) for b in recorder.batches] == [4, 4, 2]
    assert results[0].batch_size == 4 and results[-1].batch_size == 2
    assert all(r.inference_us >= 0 and r.wait_us >= 0 for r in results)


def test_a_lone_request_waits_at_most_one_window():
    recorder = _Recorder()

    async def scenario():
        batcher = predict.Batcher(recorder, max_batch_size=64, window_s=0.02)
        batcher.start()
        try:
            start = time.perf_counter()
            result = await batcher.submit(3)
            return result, time.perf_counter() - start
        finally:
            await batcher.close()

    result, elapsed = asyncio.run(scenario())
    assert result.value == 30 and result.batch_size == 1
    assert 0.015 <= elapsed < 0.5
    assert result.wait_us >= 15_000


def test_a_failing_batch_fails_only_its_own_requests():
    recorder = _Recorder(fail_on=2)

    async def scenario():
        batcher = predict.Batcher(recorder, max_batch_size=3, window_s=0.05)
        batcher.start()
        try:
            first = await _submit_all(batcher, [1, 2, 3])
            second = await _submit_all(batcher, [4, 5])
            return first, second
        finally:
            await batcher.close()

    first, second = asyncio.run(scenario())
    assert all(isinstance(r, ValueError) for r in first)
    assert [r.value for r in second] == [40, 50]


def test_requests_queue_behind_a_running_batch():
    recorder = _Recorder(delay_s=0.05)

    async def scenario():
        batcher = predict.Batcher(recorder, max_batch_size=8, window_s=0.0)
        batcher.start()
        try:
            first = asyncio.ensure_future(batcher.submit(0))
            await asyncio.sleep(0.01)  # The first batch is now running.
            rest = await _submit_all(batcher, range(1, 6))
            return await first, rest
        finally:
            await batcher.close()

    first, rest = asyncio.run(scenario())
    assert first.batch_size == 1
    assert {r.batch_size for r in rest} == {5}


def test_submit_requires_start():
    batcher = predict.Batcher(_Recorder(), max_batch_size=2, window_s=0.0)
    with pytest.raises(RuntimeError):
        asyncio.run(batcher.submit(1))


def test_invalid_configuration():
    with pytest.raises(ValueError):
        predict.Batcher(_Recorder(), max_batch_size=0, window_s=0.0)
