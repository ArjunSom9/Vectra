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

"""Repository-wide pytest configuration."""

import os
import shutil

import pytest

# TorchInductor's CPU backend compiles generated C++ with $CXX, read once when
# torch._inductor.config is first imported. A stale value (for example a
# compiler version since removed) would fail every CPU compile, so fall back
# to the platform default. This runs at import, before any test conftest.
for _variable in ('CXX', 'CC'):
    _value = os.environ.get(_variable)
    if _value and shutil.which(_value) is None:
        del os.environ[_variable]


def pytest_collection_modifyitems(config, items):
    del config  # Unused.
    gpu_items = [item for item in items if 'gpu' in item.keywords]
    if not gpu_items:
        return
    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason='needs a CUDA GPU')
    for item in gpu_items:
        item.add_marker(skip)
