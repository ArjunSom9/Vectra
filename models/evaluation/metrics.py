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

"""Accuracy metrics for price predictions, computed in dollars."""

import numpy as np


def price_errors(
    predicted_log: np.ndarray, actual_log: np.ndarray
) -> dict[str, float]:
    """Standard automated-valuation accuracy measures.

    Args:
      predicted_log: Predicted natural-log prices.
      actual_log: Realized natural-log prices.

    Returns:
      mae_usd: Mean absolute error in dollars.
      mape: Mean absolute percentage error (fraction).
      median_ape: Median absolute percentage error (fraction).
      within_10pct: Share of predictions within 10% of the sale price.
      r2_log: Coefficient of determination of the log prices.
      count: Number of predictions.
    """
    predicted_log = np.asarray(predicted_log, dtype=np.float64)
    actual_log = np.asarray(actual_log, dtype=np.float64)
    if not predicted_log.size:
        return {
            'mae_usd': 0.0,
            'mape': 0.0,
            'median_ape': 0.0,
            'within_10pct': 0.0,
            'r2_log': 0.0,
            'count': 0,
        }
    predicted = np.exp(predicted_log)
    actual = np.exp(actual_log)
    ape = np.abs(predicted - actual) / actual
    residual = np.sum((actual_log - predicted_log) ** 2)
    total = np.sum((actual_log - actual_log.mean()) ** 2)
    return {
        'mae_usd': float(np.mean(np.abs(predicted - actual))),
        'mape': float(np.mean(ape)),
        'median_ape': float(np.median(ape)),
        'within_10pct': float(np.mean(ape <= 0.10)),
        'r2_log': float(1.0 - residual / total) if total > 0 else 0.0,
        'count': int(predicted_log.size),
    }
