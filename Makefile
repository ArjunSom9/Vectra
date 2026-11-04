# Vectra: build, test, benchmark and release entry points.
#
#   make bootstrap      toolchain + Python environment (Ubuntu 24.04)
#   make proto          Python gRPC stubs into build/gen/python
#   make build          engine (with server) + Python packages
#   make test           every suite that runs on this host
#   make lint           clang-format, cpplint, ruff, pylint
#
# The engine's server layer needs gRPC 1.71.0 installed under GRPC_PREFIX
# (scripts/bootstrap.sh builds it). Without it, `make build-engine-core`
# and `make test-engine` still build and test the dependency-free core.

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

PYTHON      ?= .venv/bin/python
GRPC_PREFIX ?= /opt/vectra/grpc-1.71.0
BUILD_TYPE  ?= RelWithDebInfo
VECTRA_ARCH ?= native
JOBS        ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
CORPUS      ?= data/corpus/dev
CHECKPOINT  ?= checkpoints/price_transformer_v1
BUNDLE      ?= dist/compiled_model

ENGINE_BUILD := build/engine
CORE_BUILD   := build/engine-core
GEN_PYTHON   := build/gen/python
PYTHONPATH   := .:kernels:compiler:$(GEN_PYTHON)
export PYTHONPATH

PY_SOURCES := data kernels compiler models serving benchmark tests conftest.py
CXX_SOURCES = $(shell find engine contracts \( -name '*.cc' -o -name '*.h' \
                -o -name '*.proto' \))

.PHONY: help bootstrap lock proto build build-engine build-engine-core \
        build-python dist test test-engine test-engine-sanitizers \
        test-kernels test-compiler test-models test-serving test-e2e \
        test-regression lint format corpus train export serve cluster \
        bench-qps bench-recall bench-gpu bench-bandwidth clean

help:  ## List targets.
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk -F':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'

bootstrap:  ## Toolchain, gRPC 1.71.0 and the Python environment.
	scripts/bootstrap.sh

lock:  ## Re-resolve both lock files inside the project window.
	uv pip compile pyproject.toml --all-extras --universal \
	  --python-version 3.12 --exclude-newer 2025-05-30T23:59:59Z \
	  --no-header --annotation-style line -o requirements-lock.txt
	uv pip compile pyproject.toml --universal \
	  --python-version 3.12 --exclude-newer 2025-05-30T23:59:59Z \
	  --no-header --annotation-style line -o requirements-runtime-lock.txt

proto:  ## Python stubs for contracts/proto (C++ stubs are a CMake step).
	scripts/gen_proto.sh

# --------------------------------------------------------------------------
# Build.
# --------------------------------------------------------------------------
build: build-engine build-python  ## Everything.

build-engine:  ## Engine, server, client and tools (needs gRPC).
	cmake -S engine -B $(ENGINE_BUILD) -G Ninja \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_PREFIX_PATH=$(GRPC_PREFIX) \
	  -DVECTRA_BUILD_SERVER=ON -DVECTRA_BUILD_TESTS=ON \
	  -DVECTRA_ARCH=$(VECTRA_ARCH) -DVECTRA_ENABLE_LTO=ON
	cmake --build $(ENGINE_BUILD) -j $(JOBS)

build-engine-core:  ## The dependency-free core and its tests only.
	cmake -S engine -B $(CORE_BUILD) -G Ninja \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DVECTRA_BUILD_SERVER=OFF \
	  -DVECTRA_BUILD_TESTS=ON -DVECTRA_ARCH=$(VECTRA_ARCH)
	cmake --build $(CORE_BUILD) -j $(JOBS)

build-python: proto  ## Install the Python packages (editable).
	uv pip sync --python $(PYTHON) --exclude-newer 2025-05-30T23:59:59Z \
	  requirements-lock.txt
	uv pip install --python $(PYTHON) --no-deps -e .

dist: build export  ## Release artifacts into dist/.
	cmake --install $(ENGINE_BUILD) --prefix dist/engine
	uv build --wheel --out-dir dist

# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------
test:  ## Every suite that runs on this host.
test: test-engine test-kernels test-compiler test-models test-serving \
      test-e2e test-regression

test-engine: build-engine-core  ## C++ unit and GeoHash boundary suites.
	ctest --test-dir $(CORE_BUILD) --output-on-failure -j $(JOBS)
	if [[ -d $(ENGINE_BUILD) ]]; then \
	  ctest --test-dir $(ENGINE_BUILD) --output-on-failure -j $(JOBS); fi

test-engine-sanitizers:  ## Core suites under ASan+UBSan and TSan.
	for pair in asan:address,undefined tsan:thread; do \
	  dir=build/engine-$${pair%%:*}; \
	  cmake -S engine -B $$dir -G Ninja \
	    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVECTRA_BUILD_SERVER=OFF \
	    -DVECTRA_BUILD_TESTS=ON -DVECTRA_SANITIZE=$${pair#*:}; \
	  cmake --build $$dir -j $(JOBS); \
	  ctest --test-dir $$dir --output-on-failure -j $(JOBS); \
	done

test-kernels: proto  ## Kernel parity (Triton on GPU or under TRITON_INTERPRET=1).
	$(PYTHON) -m pytest kernels -q

test-compiler: proto  ## Pattern matching, rewrite correctness, end to end.
	$(PYTHON) -m pytest compiler/tests -q

test-models: proto  ## Dataset, model, training and the deployment compile.
	$(PYTHON) -m pytest models data -q

test-serving: proto  ## Gateway, channel pool and batching.
	$(PYTHON) -m pytest serving/tests -q

test-e2e: proto  ## Full stack; uses the C++ engine when it is built.
	VECTRA_ENGINE_BUILD=$(ENGINE_BUILD) $(PYTHON) -m pytest tests -q

test-regression:  ## Gate benchmark reports against tests/regression goldens.
	$(PYTHON) -m pytest tests/regression benchmark -q
	if ls benchmark/reports/search.json >/dev/null 2>&1; then \
	  $(PYTHON) -m tests.regression.compare_snapshots \
	    --golden=tests/regression/golden_latency.json \
	    --report=benchmark/reports/search.json \
	    $$(ls benchmark/reports/ping.json 2>/dev/null | sed 's/^/--report=/'); \
	fi
	if ls benchmark/reports/recall.json >/dev/null 2>&1; then \
	  $(PYTHON) -m tests.regression.compare_snapshots \
	    --golden=tests/regression/golden_recall.json \
	    --report=benchmark/reports/recall.json; \
	fi

# --------------------------------------------------------------------------
# Style.
# --------------------------------------------------------------------------
lint:  ## clang-format, cpplint, ruff and pylint (Google style guides).
	.venv/bin/clang-format --dry-run --Werror $(CXX_SOURCES)
	.venv/bin/cpplint --quiet $(filter-out %.proto,$(CXX_SOURCES))
	.venv/bin/ruff check $(PY_SOURCES)
	.venv/bin/ruff format --check $(PY_SOURCES)
	.venv/bin/pylint --rcfile=.pylintrc $(PY_SOURCES)

format:  ## Apply clang-format and ruff format.
	.venv/bin/clang-format -i $(CXX_SOURCES)
	.venv/bin/ruff format $(PY_SOURCES)

# --------------------------------------------------------------------------
# Data, model and serving.
# --------------------------------------------------------------------------
corpus:  ## Synthetic dev corpus, GeoHash cells and shard split.
	$(PYTHON) -m data.pipelines.generate_embeddings --out_dir=$(CORPUS)
	$(PYTHON) -m data.pipelines.assign_geohash --corpus_dir=$(CORPUS)
	$(PYTHON) -m data.pipelines.validate_corpus --corpus_dir=$(CORPUS)
	$(PYTHON) -m data.pipelines.split_shards --corpus_dir=$(CORPUS)

train:  ## Train the price model (eager).
	$(PYTHON) -m models.training.train --checkpoint_dir=$(CHECKPOINT)
	$(PYTHON) -m models.evaluation.backtest --checkpoint_dir=$(CHECKPOINT)

export:  ## Deployment compile into $(BUNDLE).
	$(PYTHON) -m models.export.compile_entry \
	  --checkpoint_dir=$(CHECKPOINT) --out=$(BUNDLE)

serve:  ## Run the inference gateway.
	$(PYTHON) -m serving.service --config=serving/configs/service.yaml

cluster:  ## Index the dev corpus and run three local shards.
	scripts/run_local_cluster.sh

# --------------------------------------------------------------------------
# Benchmarks (reports land in benchmark/reports/).
# --------------------------------------------------------------------------
bench-qps:  ## A1 + A3: open-loop soak at 50,000 QPS.
	$(PYTHON) -m benchmark.qps_latency.run_soak --mode=search --qps=50000 \
	  --duration=30m --queries=$(CORPUS)/queries \
	  --out=benchmark/reports/search.json
	$(PYTHON) -m benchmark.qps_latency.run_soak --mode=ping --qps=50000 \
	  --target=$${VECTRA_PING_TARGET:-127.0.0.1:7101} \
	  --out=benchmark/reports/ping.json

bench-recall:  ## A2 + A3: recall@10 versus ef_search, with the hnswlib oracle.
	$(PYTHON) -m benchmark.recall.run_recall_sweep --corpus_dir=$(CORPUS) \
	  --oracle --out=benchmark/reports/recall_sweep.json
	$(PYTHON) -m benchmark.recall.analyze_recall \
	  --sweep=benchmark/reports/recall_sweep.json \
	  --out=benchmark/reports/recall.json

bench-gpu:  ## B3: compiled versus eager serving throughput (GPU).
	for m in eager inductor vectra; do \
	  $(PYTHON) -m benchmark.gpu_throughput.run_throughput --mode=$$m \
	    --checkpoint_dir=$(CHECKPOINT) \
	    --out=benchmark/reports/throughput_$$m.json; \
	done
	$(PYTHON) -m benchmark.gpu_throughput.compare_runs \
	  --baseline=benchmark/reports/throughput_eager.json \
	  --candidate=benchmark/reports/throughput_vectra.json

bench-bandwidth:  ## B1 + B3 + C3: Nsight Compute byte and bandwidth proof.
	$(PYTHON) -m benchmark.bandwidth.run_bandwidth_proof \
	  --out=benchmark/reports/bandwidth.json

clean:  ## Remove build outputs (keeps the corpus and checkpoints).
	rm -rf build/engine build/engine-core build/engine-asan \
	  build/engine-tsan build/gen/python dist/compiled_model dist/engine
