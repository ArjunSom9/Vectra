# Annotated bibliography

Numbering follows the design document's Works Cited (§12); the documents
themselves are in `Bibliography/`. Every source predates the end of the
project window (May 30, 2025). Each entry states what the project uses it
for and, where it matters, what checking the source changed.

## Systems and retrieval

**[1] J. J. Pan, J. Wang, G. Li, "Survey of Vector Database Management
Systems," VLDB Journal 33(5), 2024.** The structure of managed vector
systems: partitioned collections answered by scatter-gather, with
"extra overhead for search queries due to scatter-gather," and rule- or
cost-based plan selection. Used in [02](../architecture/02-why-not-managed.md)
and ADR-001.

**[4] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks,"
Information Systems 87, 2020.** The recall protocol: recall@k against exact
ground truth, one curve point per search parameter. Implemented by
`benchmark/recall/`. The benchmark reports throughput and recall; it does
not prescribe a load-generation model. Vectra's latency is measured
open-loop (fixed arrival schedule, latency from the scheduled send time),
because a closed-loop generator under-reports a stalled server's queueing.

**[7] H. V. Simhadri et al., "Results of the NeurIPS'21 Challenge on
Billion-Scale ANN Search," PMLR 176, 2022.** In-memory graph indexes reach
microsecond-range latency only when RAM-resident; disk-resident variants
degrade under concurrency. Supports the residency argument of 02 §4.

**[11] J. Wang et al., "Milvus: A Purpose-Built Vector Data Management
System," SIGMOD 2021.** LSM-style segments, replica-based availability and
cost-based planning, built for operability across workloads. Used in 02.

**[12] S. Jayaram Subramanya et al., "DiskANN," NeurIPS 2019.** Disk-resident
graph search and its I/O behavior. Used in 02 §4.

**[13] J. Dean, L. A. Barroso, "The Tail at Scale," CACM 56(2), 2013.**
Fan-out amplifies latency variability in proportion to the subsystems a
request waits on. The basis of the bounded fan-out (ADR-002) and of 02 §3.

**[17] "Geohash," Wikipedia (encoding by G. Niemeyer, 2008).** The encoding
(bit interleaving, base-32 alphabet) and the cell-size table at the
equator (precision 4: 39.1 km × 19.5 km). `GeohashTest.EncodesReferenceExamples`
uses its examples. It is a tertiary source and sufficient for a public
encoding; cell widths at other latitudes are computed, not taken from it.

**[18] Elastic, "Geohash grid aggregation," Elasticsearch Reference 7.17.**
Cell dimensions per precision, cross-checking [17].

**[19] gRPC Authors, "Asynchronous-API tutorial — C++."** The
`CompletionQueue` loop and the `CallData` pattern (address as tag, a new
call created before the current one is processed). Implemented in
`engine/src/server/`.

**[20] M. Belshe, R. Peon, M. Thomson, "HTTP/2," RFC 7540, 2015.**
`SETTINGS_MAX_CONCURRENT_STREAMS` caps the streams of one connection, which
is why one gRPC channel cannot carry 50,000 QPS. Used in
[03](../architecture/03-async-grpc.md) §6.

**[27] gRPC Authors, "Performance Best Practices."** Re-use channels and
stubs; for high load, use a pool of channels with distinct channel
arguments (or `GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL`) so that they do not
share a connection. Both are implemented, and
`serving/tests/test_channel_pool.py` demonstrates the shared connection
that identical channels would get. For the asynchronous server the guide
recommends numcpu threads and reports that **two threads per completion
queue** gave the best performance. The design document deploys one, so
`ServerOptions.threads_per_queue` is configurable and the ratio is chosen
by benchmark.

**[21] Y. A. Malkov, D. A. Yashunin, "HNSW," TPAMI 42(4), 2020.** Level
assignment with mL = 1/ln M, Mmax0 = 2M, and the neighbor-selection
heuristic (Algorithm 4). Implemented in `engine/src/index/hnsw_*.cc`.

**[22] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
Neighbor Search," TPAMI 33(1), 2011.** Subspace quantization and asymmetric
distance computation. Implemented in `pq_codec.cc`. The paper's own
results motivate re-ranking with exact distances when recall matters
(ADR-004).

**[23] Y. Malkov et al., "hnswlib," GitHub.** The flat-array reference
implementation and the recall oracle of `make bench-recall`. The latest
release is 0.8.0 (December 2023); an earlier README's "0.9.0" never existed.

**[28] M. Wang et al., "A Comprehensive Survey and Experimental Comparison
of Graph-Based ANN Search," PVLDB 14(11), 2021.** Graph construction
trade-offs, and flat layouts in the fastest implementations.

**[29] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search with
GPUs," IEEE TBD 7(3), 2021.** Fused, tiled distance and k-selection kernels
running at large fractions of peak.

**[30] C. Aguerrebere et al., "Similarity Search in the Blink of an Eye with
Compressed Indices," PVLDB 16(11), 2023.** Compressed graph indexes and
memory bandwidth as the bottleneck of graph search.

**[31] R. Guo et al., "Accelerating Large-Scale Inference with Anisotropic
Vector Quantization," ICML 2020.** Quantization loss aligned with inner-product
ranking. A considered alternative to PQ, not adopted.

## GPUs, kernels and compilers

**[2] A. Paszke et al., "PyTorch," NeurIPS 2019.** Eager, operator-by-operator
execution. Used in [06](../architecture/06-memory-bottleneck.md).

**[3] T. Dao et al., "FlashAttention," NeurIPS 2022.** IO-aware attention:
fuse, tile, and never materialize the N×N matrix. The structure of the
fused kernel ([07](../architecture/07-triton-kernels.md)).

**[25] T. Dao, "FlashAttention-2," ICLR 2024.** Work partitioning across
the sequence dimension and fewer non-matmul FLOPs. Informs the grid and the
exp2-based online softmax.

**[5] NVIDIA, "Nsight Compute Documentation."** The metrics of the GPU gates:
`dram__bytes_read.sum`, `dram__bytes_write.sum`,
`dram__throughput.avg.pct_of_peak_sustained_elapsed`, and the cache- and
clock-control settings fixed in `kernels/benchmarks/nsight/README.md`.

**[6] S. Williams, A. Waterman, D. Patterson, "Roofline," CACM 52(4), 2009.**
Attainable performance = min(peak, intensity × bandwidth).
`kernels/benchmarks/roofline.py` places both formulations. Checking it
against the H100 numbers showed that the fused kernel stays below the
ridge point at the serving shape (06 §4).

**[8] R. Pope et al., "Efficiently Scaling Transformer Inference," MLSys
2023.** Low-batch inference is bound by memory movement.

**[14] Y.-H. H. Tsai et al., "Transformer Dissection," EMNLP-IJCNLP 2019.**
Attention as a kernel smoother. The RBF kernel exp(−‖q−k‖²/2σ²) is a valid
attention kernel, which is the model's spatial attention.

**[15] J. T. Frank et al., "Euclidean Fast Attention," arXiv:2412.08541,
December 2024.** Euclidean-distance-based attention in practice.

**[16] NVIDIA, "H100 Tensor Core GPU Architecture" whitepaper, 2022.** H100
SXM5: 132 SMs, 228 KB of shared memory per SM, 50 MB L2, 3.35 TB/s HBM3,
989.4 dense FP16 tensor TFLOP/s. The figures of
`vectra_kernels.tile_configs.H100_SXM5`.

**[26] P. Tillet, H. T. Kung, D. Cox, "Triton," MAPL 2019** and **[32]
P. Tillet, "Introducing Triton," OpenAI, 2021.** Tile-level programming with
compiler-managed layout. ADR-005.

**[33] NVIDIA, "Hopper Tuning Guide."** Shared-memory capacity and
asynchronous copy behavior used in tile sizing.

**[9] J. Reed et al., "torch.fx," MLSys 2022** and **[10] PyTorch,
"torch.fx.subgraph_rewriter.replace_pattern."** Graph capture and
pattern-based rewriting by use-def topology.
`replace_pattern_with_filters` (`match_filters`, `ignore_literals`,
`replacement_callback`) is the API the pass uses ([09](../architecture/09-subgraph-rewrite.md)).

**[24] J. Ansel et al., "PyTorch 2," ASPLOS 2024** and **[36] PyTorch
Foundation, "PyTorch 2.0" release post, 2023.** TorchDynamo, AOTAutograd
and TorchInductor ([08](../architecture/08-fx-compilation.md)).

**[34] B. Cannon and D. Viehland, "PEP 523 — Adding a Frame Evaluation API
to CPython," 2016.** The hook TorchDynamo uses. The design document
attributes the PEP to "V. Stinner and N. Coghlan"; its authors are Brett
Cannon and Dino Viehland.

**[35] PyTorch, "torch.cond."** Data-dependent control flow without a graph
break, used by `trim_to_radius`. The documentation marks `torch.cond` as a
**prototype** feature. That is a risk the zero-break test in
`models/tests/test_model.py` guards against on every PyTorch upgrade.
