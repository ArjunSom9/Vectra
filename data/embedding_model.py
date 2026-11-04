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

"""The property embedding: hedonic attributes and location -> 128 floats.

The corpus pipeline and the gateway embed properties with this one function,
so query embeddings and corpus embeddings live in the same space. The map is
a fixed, seeded two-layer network over standardized attributes and
multi-scale positional features; positional features make nearby properties
near in embedding space, which keeps radius-filtered graph search efficient.
The sale price is deliberately not an input: it is the model's target.

Typical usage:

  embedder = embedding_model.PropertyEmbedder()
  vectors = embedder.embed(attributes, latitude, longitude)  # [n, 128] f32
"""

from collections.abc import Mapping
import math

import numpy as np

EMBEDDING_DIM = 128
_SEED = 20250106
_HIDDEN = 256
# Positional wavelengths, in kilometers: regional to block level.
_WAVELENGTHS_KM = (400.0, 100.0, 25.0, 6.0, 1.5)
# Reference latitude of the equirectangular projection (continental US).
_REFERENCE_LAT = 37.0
_REFERENCE_YEAR = 2025

ATTRIBUTE_COLUMNS = (
    'living_area_sqft',
    'lot_area_sqft',
    'bedrooms',
    'bathrooms',
    'year_built',
)


def attribute_features(attributes: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standardized hedonic features, shape [n, 5], float32."""
    living = np.asarray(attributes['living_area_sqft'], dtype=np.float64)
    lot = np.asarray(attributes['lot_area_sqft'], dtype=np.float64)
    beds = np.asarray(attributes['bedrooms'], dtype=np.float64)
    baths = np.asarray(attributes['bathrooms'], dtype=np.float64)
    year = np.asarray(attributes['year_built'], dtype=np.float64)
    columns = [
        (np.log(np.maximum(living, 100.0)) - 7.45) / 0.45,
        (np.log(np.maximum(lot, 100.0)) - 8.6) / 0.9,
        (beds - 3.0) / 1.2,
        (baths - 2.2) / 0.9,
        ((_REFERENCE_YEAR - year) - 40.0) / 25.0,
    ]
    return np.stack(columns, axis=-1).astype(np.float32)


def positional_features(
    latitude: np.ndarray, longitude: np.ndarray
) -> np.ndarray:
    """Multi-scale sinusoidal position features, shape [n, 20], float32."""
    lat = np.asarray(latitude, dtype=np.float64)
    lng = np.asarray(longitude, dtype=np.float64)
    km_per_degree = math.pi / 180.0 * 6371.0088
    east = lng * km_per_degree * math.cos(math.radians(_REFERENCE_LAT))
    north = lat * km_per_degree
    columns = []
    for wavelength in _WAVELENGTHS_KM:
        omega = 2.0 * math.pi / wavelength
        for axis in (east, north):
            columns.append(np.sin(omega * axis))
            columns.append(np.cos(omega * axis))
    return np.stack(columns, axis=-1).astype(np.float32)


class PropertyEmbedder:
    """The fixed embedding network. Deterministic for a given seed.

    Attributes:
      dimension: Output dimension.
    """

    def __init__(self, seed: int = _SEED, dimension: int = EMBEDDING_DIM):
        """Draws the network weights from a seeded generator.

        Args:
          seed: Generator seed; corpus and gateway must use the same value.
          dimension: Embedding dimension.
        """
        self.dimension = dimension
        rng = np.random.default_rng(seed)
        inputs = 5 + 4 * len(_WAVELENGTHS_KM)
        self._w1 = (
            rng.standard_normal((inputs, _HIDDEN)) / math.sqrt(inputs)
        ).astype(np.float32)
        self._b1 = (0.1 * rng.standard_normal(_HIDDEN)).astype(np.float32)
        self._w2 = (
            rng.standard_normal((_HIDDEN, dimension)) / math.sqrt(_HIDDEN)
        ).astype(np.float32)

    def embed(
        self,
        attributes: Mapping[str, np.ndarray],
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:
        """Embeds properties.

        Args:
          attributes: Columns named in ATTRIBUTE_COLUMNS, each of shape [n].
          latitude: Shape [n], degrees.
          longitude: Shape [n], degrees.

        Returns:
          Float32 array of shape [n, dimension].
        """
        features = np.concatenate(
            [
                attribute_features(attributes),
                1.5 * positional_features(latitude, longitude),
            ],
            axis=-1,
        )
        hidden = np.maximum(features @ self._w1 + self._b1, 0.0)
        return np.tanh(hidden @ self._w2).astype(np.float32)
