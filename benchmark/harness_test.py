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

"""The benchmark harnesses: gates, parsing and the recall sweep itself."""

import asyncio
import csv
import io

import numpy as np
import pytest
import yaml

from benchmark.bandwidth import collect_nsight_metrics
from benchmark.bandwidth import run_bandwidth_proof
from benchmark.gpu_throughput import compare_runs
from benchmark.qps_latency import percentiles
from benchmark.recall import analyze_recall
from benchmark.recall import run_recall_sweep
from data import embedding_model
from data import routing
from data import synthetic
from data.pipelines import assign_geohash
from data.pipelines import corpus_io
from data.pipelines import generate_embeddings
from kernels.benchmarks import roofline
from serving.tests import fake_engine


def _run(qps, p50, achieved=None, errors=0):
    return {
        'offered_qps': qps,
        'achieved_qps': achieved or qps,
        'p50_us': p50,
        'errors': errors,
    }


def test_max_sustained_qps_stops_at_the_first_failure():
    runs = [
        _run(10_000, 300),
        _run(30_000, 450),
        _run(50_000, 800),
        _run(60_000, 1400),
        _run(70_000, 900),
    ]
    assert percentiles.max_sustained_qps(runs, 'p50_us', 1000) == 50_000
    undelivered = [_run(50_000, 500, achieved=45_000)]
    assert percentiles.max_sustained_qps(undelivered, 'p50_us', 1000) is None
    assert '50,000' in percentiles.table(runs)


def test_summarize_latencies():
    summary = percentiles.summarize(np.arange(1, 1001))
    assert summary['count'] == 1000
    assert summary['p50_us'] == pytest.approx(500.5)
    assert summary['max_us'] == 1000


def test_recall_analysis_reads_the_configured_operating_point():
    sweep = {
        'configured_ef_search': 128,
        'boundary_recall': 1.0,
        'boundary_queries': 12,
        'curve': [
            {'ef_search': 64, 'recall_at_10': 0.93},
            {'ef_search': 128, 'recall_at_10': 0.97},
            {'ef_search': 256, 'recall_at_10': 0.99},
        ],
    }
    report = analyze_recall.summarize(sweep, target=0.95)
    assert report['recall_at_10'] == 0.97
    assert report['min_ef_search_for_target'] == 128
    with pytest.raises(ValueError):
        analyze_recall.summarize(dict(sweep, configured_ef_search=512), 0.95)


def test_throughput_gate():
    eager = {
        'mode': 'eager',
        'device': 'H100',
        'batch_size': 64,
        'block_dtype': 'float16',
        'num_candidates': 64,
        'requests_per_s': 100_000.0,
        'max_abs_error_vs_eager': 0.0,
    }
    fast = dict(
        eager,
        mode='vectra',
        requests_per_s=340_000.0,
        max_abs_error_vs_eager=0.004,
        numerics_rtol=1e-3,
    )
    passed, problems, ratio = compare_runs.compare(eager, fast, 3.0)
    assert passed and not problems and ratio == pytest.approx(3.4)
    wrong = dict(fast, max_abs_error_vs_eager=0.5)
    assert not compare_runs.compare(eager, wrong, 3.0)[0]
    other_batch = dict(fast, batch_size=128)
    assert not compare_runs.compare(eager, other_batch, 3.0)[0]


_NCU_HEADER = (
    'ID',
    'Process ID',
    'Process Name',
    'Host Name',
    'Kernel Name',
    'Context',
    'Stream',
    'Block Size',
    'Grid Size',
    'Device',
    'CC',
    'Section Name',
    'Metric Name',
    'Metric Unit',
    'Metric Value',
)
# (kernel id, kernel name, metric, unit, value) as `ncu --csv` prints them.
_NCU_ROWS = (
    (
        '0',
        '_rbf_attention_kernel',
        'dram__bytes_read.sum',
        'byte',
        '50,331,648',
    ),
    (
        '0',
        '_rbf_attention_kernel',
        'dram__bytes_write.sum',
        'byte',
        '16,777,216',
    ),
    (
        '0',
        '_rbf_attention_kernel',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        '%',
        '70',
    ),
    (
        '0',
        '_rbf_attention_kernel',
        'gpu__time_duration.sum',
        'nsecond',
        '30,000',
    ),
    ('1', '_squared_norms_kernel', 'dram__bytes_read.sum', 'byte', '1,048,576'),
    ('1', '_squared_norms_kernel', 'dram__bytes_write.sum', 'byte', '0'),
    (
        '1',
        '_squared_norms_kernel',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        '%',
        '40',
    ),
    (
        '1',
        '_squared_norms_kernel',
        'gpu__time_duration.sum',
        'nsecond',
        '10,000',
    ),
)


def _ncu_csv() -> str:
    out = io.StringIO()
    out.write('==PROF== Connected to process 4242\n')
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator='\n')
    writer.writerow(_NCU_HEADER)
    for kernel, name, metric, unit, value in _NCU_ROWS:
        writer.writerow(
            (
                kernel,
                '4242',
                'python',
                'host',
                name,
                '1',
                '7',
                '(128,1,1)',
                '(1,4096,1)',
                '0',
                '9.0',
                'Command line profiler metrics',
                metric,
                unit,
                value,
            )
        )
    return out.getvalue()


def test_nsight_csv_is_aggregated_over_the_region():
    totals = collect_nsight_metrics.aggregate(
        collect_nsight_metrics.parse_csv(_ncu_csv())
    )
    assert totals['kernels'] == 2
    assert totals['dram_bytes'] == 50_331_648 + 16_777_216 + 1_048_576
    assert totals['gpu_time_ns'] == 40_000
    # Time-weighted: (70 * 30 + 40 * 10) / 40.
    assert totals['dram_throughput_pct'] == pytest.approx(62.5)
    command = collect_nsight_metrics.ncu_command(['python', 'x.py'], 'r')
    assert '--cache-control' in command and 'r/' in command


def test_bandwidth_gate():
    with open('serving/configs/slo.yaml', encoding='utf-8') as f:
        slo = yaml.safe_load(f)
    good = {
        64: {
            'eager': {'dram_bytes': 300e6, 'dram_throughput_pct': 35.0},
            'fused': {'dram_bytes': 5e6, 'dram_throughput_pct': 20.0},
        },
        1024: {
            'eager': {'dram_bytes': 4.8e9, 'dram_throughput_pct': 41.0},
            'fused': {'dram_bytes': 70e6, 'dram_throughput_pct': 71.0},
        },
    }
    assert not run_bandwidth_proof.evaluate(good, slo)
    slow = {
        64: good[64],
        1024: dict(
            good[1024], fused={'dram_bytes': 70e6, 'dram_throughput_pct': 50.0}
        ),
    }
    assert len(run_bandwidth_proof.evaluate(slow, slo)) == 1


def test_traffic_model_at_the_serving_shape():
    budget = roofline.analyze(roofline.SERVING)
    # The fused kernel moves under 2% of the eager chain's bytes ...
    assert budget['bytes_ratio'] < 0.02
    # ... but stays below the H100 ridge point: still bandwidth-bound.
    assert budget['fused_is_memory_bound']
    assert budget['fused_working_set_fits_l2']
    big = roofline.analyze(roofline.Shape(1024, 4, 65, 65, 32, 32, 2))
    assert not big['fused_working_set_fits_l2']


def _corpus_dir(tmp_path) -> str:
    everything = synthetic.generate(3060, seed=5)
    corpus, queries = (
        everything.take(np.arange(3000)),
        everything.take(np.arange(3000, 3060)),
    )
    directory = str(tmp_path / 'corpus')
    embedder = embedding_model.PropertyEmbedder()
    generate_embeddings.write_corpus(directory, corpus, embedder, 4096)
    corpus_io.write_column(
        directory,
        corpus_io.GEOHASH_COLUMN,
        assign_geohash.assign(corpus.latitude, corpus.longitude),
    )
    generate_embeddings.write_corpus(
        f'{directory}/queries', queries, embedder, 4096, prefix='query_'
    )
    return directory


def test_recall_sweep_is_perfect_against_exact_shards(tmp_path):
    directory = _corpus_dir(tmp_path)
    config = routing.load_engine_config('engine/configs')
    config.topology.client.timeout_ms = 5000
    truth = run_recall_sweep.GroundTruth(directory)
    q_dir = f'{directory}/queries'
    embeddings = corpus_io.read_column(q_dir, 'query_embedding')
    lats = corpus_io.read_column(q_dir, 'query_latitude')
    lngs = corpus_io.read_column(q_dir, 'query_longitude')
    queries = [
        (embeddings[i], float(lats[i]), float(lngs[i])) for i in range(30)
    ]
    truths = [truth.search(e, lat, lng, 5.0, 10) for e, lat, lng in queries]

    async def scenario():
        cluster = await fake_engine.start_cluster(
            fake_engine.Corpus.from_dir(directory), config
        )
        try:
            return await run_recall_sweep.sweep_engine(
                queries,
                truths,
                [0, 1],
                config,
                cluster.endpoints,
                [16, 64],
                5.0,
                10,
            )
        finally:
            await cluster.stop()

    result = asyncio.run(scenario())
    assert [p['recall_at_10'] for p in result['curve']] == [1.0, 1.0]
    assert result['boundary_recall'] == 1.0


def test_global_knn_matches_sorting():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 8))
    vectors = rng.standard_normal((5, 8))
    want = np.argsort(((vectors[:, None] - data[None]) ** 2).sum(-1), 1)[:, :7]
    got = run_recall_sweep.global_knn(data, vectors, 7, block=128)
    np.testing.assert_array_equal(got, want)


def test_endpoint_overrides():
    assert run_recall_sweep.parse_endpoints(['1=a:1', '1=b:2', '3=c:3']) == {
        1: ['a:1', 'b:2'],
        3: ['c:3'],
    }
    assert run_recall_sweep.parse_endpoints([]) is None
