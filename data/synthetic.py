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

"""A seeded synthetic stand-in for the production property corpus.

No real transaction records ship with the repository. This generator draws
properties around eighteen U.S. metros (dense cores plus suburbs), hedonic
attributes, and sale prices from a hedonic model whose location premium is a
smooth random field at 1-20 km scales: the neighborhood signal that
comparable properties reveal and that the pricing model must learn from its
retrieved candidates. The same seed always yields the same corpus.
"""

import dataclasses
import math

import numpy as np

# 2015-01-01 and 2024-12-31 as days since the Unix epoch.
FIRST_SALE_DAY = 16436
LAST_SALE_DAY = 20088
_DAYS_PER_YEAR = 365.25
_ANNUAL_APPRECIATION = 0.04
_NOISE_SIGMA = 0.12


@dataclasses.dataclass(frozen=True)
class Metro:
    """A metro area: center, relative housing stock, median price."""

    name: str
    lat: float
    lng: float
    weight: float
    median_price_usd: float


METROS = (
    Metro('new_york', 40.7128, -74.0060, 20.0, 700_000),
    Metro('los_angeles', 34.0522, -118.2437, 13.0, 950_000),
    Metro('chicago', 41.8781, -87.6298, 9.0, 330_000),
    Metro('dallas', 32.7767, -96.7970, 7.0, 380_000),
    Metro('houston', 29.7604, -95.3698, 7.0, 310_000),
    Metro('washington', 38.9072, -77.0369, 6.0, 600_000),
    Metro('miami', 25.7617, -80.1918, 6.0, 550_000),
    Metro('atlanta', 33.7490, -84.3880, 6.0, 390_000),
    Metro('boston', 42.3601, -71.0589, 5.0, 750_000),
    Metro('phoenix', 33.4484, -112.0740, 5.0, 430_000),
    Metro('san_francisco', 37.7749, -122.4194, 5.0, 1_300_000),
    Metro('seattle', 47.6062, -122.3321, 4.0, 800_000),
    Metro('minneapolis', 44.9778, -93.2650, 4.0, 360_000),
    Metro('san_diego', 32.7157, -117.1611, 3.0, 900_000),
    Metro('denver', 39.7392, -104.9903, 3.0, 580_000),
    Metro('portland', 45.5152, -122.6784, 2.5, 530_000),
    Metro('austin', 30.2672, -97.7431, 2.5, 520_000),
    Metro('san_jose', 37.3382, -121.8863, 2.0, 1_400_000),
)


@dataclasses.dataclass
class Corpus:
    """Column-oriented property records (all arrays share length n)."""

    property_id: np.ndarray  # int64
    latitude: np.ndarray  # float64
    longitude: np.ndarray  # float64
    living_area_sqft: np.ndarray  # float32
    lot_area_sqft: np.ndarray  # float32
    bedrooms: np.ndarray  # int32
    bathrooms: np.ndarray  # float32
    year_built: np.ndarray  # int32
    last_sale_price_usd: np.ndarray  # float64
    last_sale_epoch_day: np.ndarray  # int32
    metro_index: np.ndarray  # int16

    def __len__(self) -> int:
        return len(self.property_id)

    def attributes(self) -> dict[str, np.ndarray]:
        """The embedding inputs (see embedding_model.ATTRIBUTE_COLUMNS)."""
        return {
            'living_area_sqft': self.living_area_sqft,
            'lot_area_sqft': self.lot_area_sqft,
            'bedrooms': self.bedrooms,
            'bathrooms': self.bathrooms,
            'year_built': self.year_built,
        }

    def take(self, rows: np.ndarray) -> 'Corpus':
        """The subset of records at `rows`."""
        return Corpus(
            **{
                field.name: getattr(self, field.name)[rows]
                for field in dataclasses.fields(self)
            }
        )


class _LocationField:
    """A smooth random field over the plane: a sum of plane waves."""

    def __init__(
        self,
        rng: np.random.Generator,
        waves: int,
        min_wavelength_km: float,
        max_wavelength_km: float,
        amplitude: float,
    ):
        wavelengths = np.exp(
            rng.uniform(
                math.log(min_wavelength_km), math.log(max_wavelength_km), waves
            )
        )
        angles = rng.uniform(0.0, 2.0 * math.pi, waves)
        self._kx = 2.0 * math.pi / wavelengths * np.cos(angles)
        self._ky = 2.0 * math.pi / wavelengths * np.sin(angles)
        self._phase = rng.uniform(0.0, 2.0 * math.pi, waves)
        self._amplitude = amplitude * math.sqrt(2.0 / waves)

    def __call__(self, east_km: np.ndarray, north_km: np.ndarray) -> np.ndarray:
        total = np.zeros(np.shape(east_km))
        for kx, ky, phase in zip(self._kx, self._ky, self._phase, strict=True):
            total += np.sin(kx * east_km + ky * north_km + phase)
        return self._amplitude * total


def _to_km(lat: np.ndarray, lng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = math.pi / 180.0 * 6371.0088
    return lng * scale * np.cos(np.radians(lat)), lat * scale


def log_price(corpus: Corpus, field: _LocationField) -> np.ndarray:
    """Noise-free hedonic log price of each record."""
    medians = np.log([m.median_price_usd for m in METROS])[corpus.metro_index]
    years = (corpus.last_sale_epoch_day - LAST_SALE_DAY) / _DAYS_PER_YEAR
    east, north = _to_km(corpus.latitude, corpus.longitude)
    return (
        medians
        + 0.75 * np.log(corpus.living_area_sqft / 1800.0)
        + 0.08 * np.log(corpus.lot_area_sqft / 6000.0)
        + 0.03 * (corpus.bedrooms - 3)
        + 0.05 * (corpus.bathrooms - 2.0)
        - 0.002 * (2025 - corpus.year_built)
        + _ANNUAL_APPRECIATION * years
        + field(east, north)
    )


def generate(num_properties: int, seed: int) -> Corpus:
    """Draws a corpus of `num_properties` records.

    Args:
      num_properties: Number of records.
      seed: Generator seed.

    Returns:
      The corpus, with property ids 1..num_properties.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([m.weight for m in METROS])
    metro = rng.choice(
        len(METROS), size=num_properties, p=weights / weights.sum()
    ).astype(np.int16)
    # 70% dense core (sigma 6 km), 30% suburbs (sigma 20 km).
    sigma = np.where(rng.random(num_properties) < 0.7, 6.0, 20.0)
    distance = np.minimum(
        np.abs(rng.normal(0.0, 1.0, num_properties)) * sigma * math.sqrt(2.0),
        60.0,
    )
    bearing = rng.uniform(0.0, 2.0 * math.pi, num_properties)
    center_lat = np.array([m.lat for m in METROS])[metro]
    center_lng = np.array([m.lng for m in METROS])[metro]
    latitude = center_lat + distance * np.cos(bearing) / 111.195
    longitude = center_lng + distance * np.sin(bearing) / (
        111.195 * np.cos(np.radians(center_lat))
    )

    living = np.clip(
        rng.lognormal(math.log(1800.0), 0.38, num_properties), 450.0, 9000.0
    )
    bedrooms = np.clip(
        np.round(living / 650.0 + rng.normal(0.0, 0.6, num_properties)), 1, 7
    )
    bathrooms = np.clip(
        np.round(
            2.0 * (0.6 + bedrooms * 0.55 + rng.normal(0.0, 0.4, num_properties))
        )
        / 2.0,
        1.0,
        6.0,
    )
    lot = np.clip(
        rng.lognormal(math.log(6000.0), 0.8, num_properties)
        * (1.0 + distance / 30.0),
        800.0,
        200_000.0,
    )
    year_built = np.clip(
        np.round(rng.normal(1978.0, 24.0, num_properties)), 1900, 2024
    )
    sale_day = rng.integers(FIRST_SALE_DAY, LAST_SALE_DAY + 1, num_properties)

    corpus = Corpus(
        property_id=np.arange(1, num_properties + 1, dtype=np.int64),
        latitude=latitude.astype(np.float64),
        longitude=longitude.astype(np.float64),
        living_area_sqft=living.astype(np.float32),
        lot_area_sqft=lot.astype(np.float32),
        bedrooms=bedrooms.astype(np.int32),
        bathrooms=bathrooms.astype(np.float32),
        year_built=year_built.astype(np.int32),
        last_sale_price_usd=np.zeros(num_properties),
        last_sale_epoch_day=sale_day.astype(np.int32),
        metro_index=metro,
    )
    field = location_field(seed)
    noise = rng.normal(0.0, _NOISE_SIGMA, num_properties)
    corpus.last_sale_price_usd = np.round(
        np.exp(log_price(corpus, field) + noise), -2
    )
    return corpus


def location_field(seed: int) -> _LocationField:
    """The neighborhood premium field used by `generate(seed=seed)`."""
    return _LocationField(
        np.random.default_rng(seed + 1),
        waves=48,
        min_wavelength_km=1.0,
        max_wavelength_km=20.0,
        amplitude=0.25,
    )
