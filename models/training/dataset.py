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

"""Training examples: subject sales and their retrieved comparables.

Each example is a property sale (the subject, whose log price is the target)
with up to K comparables chosen exactly as the serving path chooses them:
properties within the search radius, ranked by embedding distance. Only
sales recorded before the subject's sale are eligible, so no example can see
its own future; the train/validation split is by sale date for the same
reason.
"""

import dataclasses
import math

import numpy as np
import torch
import yaml

from data import embedding_model
from data import geo
from data.pipelines import corpus_io
from models.price_transformer import model as model_lib


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """How examples are drawn (models/training/configs/dataset_v1.yaml)."""

    corpus_dir: str = 'data/corpus/dev'
    radius_km: float = 5.0
    num_candidates: int = 64
    max_subjects: int = 50_000
    # Sales on or after this day (2024-01-01) form the validation set.
    split_day: int = 19_723
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> 'DatasetConfig':
        with open(path, encoding='utf-8') as f:
            values = yaml.safe_load(f)['dataset']
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


@dataclasses.dataclass
class Examples:
    """Arrays in PricingInputs order plus the target, all numpy.

    Attributes:
      inputs: Model inputs, one row per example.
      target_log_price: [N] realized natural-log sale price of the subject.
      sale_day: [N] subject sale day (days since the epoch).
      subject_row: [N] row of the subject in the corpus it was drawn from.
    """

    inputs: model_lib.PricingInputs
    target_log_price: np.ndarray
    sale_day: np.ndarray
    subject_row: np.ndarray

    def __len__(self) -> int:
        return len(self.target_log_price)

    def tensors(self, rows: np.ndarray | slice | None = None):
        """(PricingInputs of tensors, target tensor) for `rows`."""
        rows = slice(None) if rows is None else rows
        inputs = model_lib.PricingInputs(
            *(
                torch.from_numpy(np.ascontiguousarray(a[rows]))
                for a in self.inputs
            )
        )
        return inputs, torch.from_numpy(self.target_log_price[rows])

    def save(self, path: str) -> None:
        np.savez(
            path,
            target_log_price=self.target_log_price,
            sale_day=self.sale_day,
            subject_row=self.subject_row,
            **self.inputs._asdict(),
        )

    @classmethod
    def load(cls, path: str) -> 'Examples':
        with np.load(path) as archive:
            inputs = model_lib.PricingInputs(
                *(archive[name] for name in model_lib.PricingInputs._fields)
            )
            return cls(
                inputs,
                archive['target_log_price'],
                archive['sale_day'],
                archive['subject_row'],
            )


class _Corpus:
    """The corpus columns needed to assemble examples."""

    def __init__(self, directory: str):
        def read(name):
            return corpus_io.read_column(directory, name)

        self.latitude = read('latitude')
        self.longitude = read('longitude')
        self.cell6 = read(corpus_io.GEOHASH_COLUMN).astype(np.int64)
        self.log_price = np.log(read('last_sale_price_usd'))
        self.sale_day = read('last_sale_epoch_day')
        self.embeddings = corpus_io.read_column(
            directory, corpus_io.EMBEDDING_COLUMN, mmap=True
        )
        self.attributes = embedding_model.attribute_features(
            {name: read(name) for name in embedding_model.ATTRIBUTE_COLUMNS}
        )
        # Rows of each 5-character prefilter cell, as the engine indexes them.
        cell5 = self.cell6 >> 5
        order = np.argsort(cell5, kind='stable')
        keys, starts = np.unique(cell5[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        self.buckets = {
            int(key): order[start:end]
            for key, start, end in zip(keys, starts, ends, strict=True)
        }


def _comparables(
    corpus: _Corpus, subject: int, radius_km: float, k: int
) -> np.ndarray:
    """Up to k comparable rows for `subject`, nearest in embedding first."""
    lat, lng = corpus.latitude[subject], corpus.longitude[subject]
    cells = geo.cover_disk(lat, lng, radius_km, 5, max_rings=16)
    pools = [corpus.buckets[c] for c in cells if c in corpus.buckets]
    if not pools:
        return np.empty(0, dtype=np.int64)
    rows = np.concatenate(pools)
    rows = rows[
        (rows != subject) & (corpus.sale_day[rows] < corpus.sale_day[subject])
    ]
    rows = rows[
        geo.haversine_km_array(
            lat, lng, corpus.latitude[rows], corpus.longitude[rows]
        )
        <= radius_km
    ]
    if rows.size == 0:
        return rows
    distance = np.sum(
        (
            np.asarray(corpus.embeddings[rows])
            - np.asarray(corpus.embeddings[subject])
        )
        ** 2,
        axis=1,
    )
    take = min(k, len(rows))
    nearest = np.argpartition(distance, take - 1)[:take]
    return rows[nearest[np.argsort(distance[nearest], kind='stable')]]


def build_examples(config: DatasetConfig) -> Examples:
    """Draws subjects and assembles their comparable sets.

    Subjects without any eligible comparable are skipped.
    """
    corpus = _Corpus(config.corpus_dir)
    rng = np.random.default_rng(config.seed)
    subjects = rng.permutation(len(corpus.latitude))[: config.max_subjects]
    k = config.num_candidates
    kept, comparables = [], []
    for subject in subjects:
        rows = _comparables(corpus, int(subject), config.radius_km, k)
        if len(rows):
            kept.append(subject)
            comparables.append(rows)
    n = len(kept)
    index = np.zeros((n, k), dtype=np.int64)
    valid = np.zeros((n, k), dtype=bool)
    for i, rows in enumerate(comparables):
        index[i, : len(rows)] = rows
        valid[i, : len(rows)] = True
    kept = np.asarray(kept, dtype=np.int64)
    location = np.stack([corpus.latitude, corpus.longitude], axis=-1).astype(
        np.float32
    )
    inputs = model_lib.PricingInputs(
        query_embedding=np.asarray(corpus.embeddings[kept], dtype=np.float32),
        query_attributes=corpus.attributes[kept],
        query_location=location[kept],
        query_cell6=corpus.cell6[kept],
        valuation_day=corpus.sale_day[kept].astype(np.float32),
        radius_km=np.full(n, config.radius_km, dtype=np.float32),
        cand_embedding=np.asarray(
            corpus.embeddings[index.reshape(-1)], dtype=np.float32
        ).reshape(n, k, -1),
        cand_attributes=corpus.attributes[index],
        cand_log_price=corpus.log_price[index].astype(np.float32),
        cand_sale_day=corpus.sale_day[index].astype(np.float32),
        cand_location=location[index],
        cand_cell6=corpus.cell6[index],
        cand_valid=valid,
    )
    return Examples(
        inputs,
        corpus.log_price[kept].astype(np.float32),
        corpus.sale_day[kept],
        kept,
    )


def split_by_day(
    examples: Examples, split_day: int
) -> tuple[np.ndarray, np.ndarray]:
    """(train rows, validation rows): sales before / on-or-after split_day."""
    late = examples.sale_day >= split_day
    return np.flatnonzero(~late), np.flatnonzero(late)


def baseline_log_price(examples: Examples) -> np.ndarray:
    """Mean comparable log price per example (the model's starting point)."""
    weights = examples.inputs.cand_valid.astype(np.float64)
    return (examples.inputs.cand_log_price * weights).sum(-1) / np.maximum(
        weights.sum(-1), 1.0
    )


def num_batches(count: int, batch_size: int) -> int:
    return math.ceil(count / batch_size)
