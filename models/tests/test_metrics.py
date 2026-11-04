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

"""Valuation accuracy metrics."""

import numpy as np
import pytest

from models.evaluation import metrics


def test_price_errors_in_dollars():
    actual = np.log([100_000.0, 200_000.0, 400_000.0, 800_000.0])
    predicted = np.log([109_000.0, 182_000.0, 400_000.0, 1_000_000.0])
    report = metrics.price_errors(predicted, actual)
    assert report['count'] == 4
    assert report['mae_usd'] == pytest.approx(
        (9_000 + 18_000 + 0 + 200_000) / 4
    )
    assert report['mape'] == pytest.approx((0.09 + 0.09 + 0.0 + 0.25) / 4)
    assert report['median_ape'] == pytest.approx(0.09)
    assert report['within_10pct'] == pytest.approx(0.75)
    assert report['r2_log'] < 1.0


def test_perfect_predictions():
    actual = np.log([150_000.0, 650_000.0])
    report = metrics.price_errors(actual, actual)
    assert report['mape'] == 0.0
    assert report['r2_log'] == pytest.approx(1.0)


def test_empty_input():
    assert metrics.price_errors(np.array([]), np.array([]))['count'] == 0
