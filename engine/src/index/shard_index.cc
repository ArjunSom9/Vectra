// Copyright 2025 Texas A&M University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vectra/index/shard_index.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vectra/index/simd_distance.h"
#include "vectra/io/binary_file.h"
#include "vectra/routing/geohash_neighbors.h"

namespace vectra::index {
namespace {

// Superseded prefilters are freed only after this long; no search runs
// anywhere near this long (the gateway deadline is milliseconds).
constexpr auto kFilterGracePeriod = std::chrono::seconds(60);

// Cap on prefilter rings: 64 rings of ~4.9 km cells, far beyond the routing
// policy's maximum radius.
constexpr int kMaxPrefilterRings = 64;

// Filtered graph searches that come back short fall back to an exhaustive scan
// when the radius population is at most this multiple of the threshold.
constexpr size_t kFallbackScanFactor = 8;

constexpr char kMagic[8] = {'V', 'E', 'C', 'T', 'R', 'A', 'I', 'X'};
constexpr uint32_t kFormatVersion = 1;

enum SectionTag : uint32_t {
  kTagHeader = 1,
  kTagVectors = 2,
  kTagCodebooks = 3,
  kTagIds = 4,
  kTagLocations = 5,
  kTagAttributes = 6,
  kTagLevels = 7,
  kTagArena = 8,
};

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t dimension;
  uint64_t count;
  uint64_t capacity;
  int32_t m;
  int32_t m0;
  int32_t ef_construction;
  int32_t ef_search;
  int32_t rerank_depth;
  int32_t pq_subspaces;
  int32_t pq_kmeans_iterations;
  int32_t build_threads;
  uint64_t pq_training_samples;
  uint64_t exhaustive_scan_threshold;
  uint64_t max_elements;
  uint64_t seed;
  uint32_t entry_node;
  int32_t entry_level;
};
static_assert(std::is_trivially_copyable_v<FileHeader>);
static_assert(std::is_trivially_copyable_v<PropertyAttributes>);
static_assert(std::is_trivially_copyable_v<routing::LatLng>);

bool Fail(std::string message, std::string* error) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

std::string ValidateOptions(const IndexOptions& options) {
  if (options.dimension <= 0) return "dimension must be positive";
  if (options.pq_subspaces <= 0 ||
      options.dimension % options.pq_subspaces != 0) {
    return "dimension must be a multiple of pq_subspaces";
  }
  if (options.m < 2 || options.m0 < options.m) {
    return "HNSW requires m >= 2 and m0 >= m";
  }
  if (options.ef_construction < 1 || options.ef_search < 1 ||
      options.rerank_depth < 1) {
    return "ef_construction, ef_search and rerank_depth must be positive";
  }
  return "";
}

// Keeps the `k` nearest of a stream of candidates in a bounded max-heap.
void PushBounded(std::vector<Neighbor>* heap, Neighbor candidate, size_t k) {
  if (heap->size() < k) {
    heap->push_back(candidate);
    std::push_heap(heap->begin(), heap->end(), CloserFirst());
  } else if (CloserFirst()(candidate, heap->front())) {
    std::pop_heap(heap->begin(), heap->end(), CloserFirst());
    heap->back() = candidate;
    std::push_heap(heap->begin(), heap->end(), CloserFirst());
  }
}

}  // namespace

ShardIndex::ShardIndex(IndexOptions options, FlatStore vectors,
                       PropertyStore properties, ProductQuantizer pq,
                       HnswGraph graph)
    : options_(std::move(options)),
      vectors_(std::move(vectors)),
      properties_(std::move(properties)),
      pq_(std::move(pq)),
      graph_(std::move(graph)),
      searcher_(&graph_, &vectors_, &pq_),
      builder_(std::make_unique<HnswBuilder>(&graph_, &vectors_,
                                             options_.ef_construction)) {}

ShardIndex::~ShardIndex() = default;

std::unique_ptr<ShardIndex> ShardIndex::Build(const IndexOptions& options,
                                              const float* embeddings,
                                              const PropertyRecord* records,
                                              size_t count,
                                              std::string* error) {
  if (const std::string problem = ValidateOptions(options); !problem.empty()) {
    Fail(problem, error);
    return nullptr;
  }
  if (count == 0) {
    Fail("cannot build an index from zero records", error);
    return nullptr;
  }
  const size_t capacity = std::max(count, options.max_elements);
  std::optional<FlatStore> vectors =
      FlatStore::Create(options.dimension, capacity);
  if (!vectors.has_value()) {
    Fail("cannot allocate the vector store", error);
    return nullptr;
  }
  PqTrainOptions pq_options;
  pq_options.num_subspaces = options.pq_subspaces;
  pq_options.kmeans_iterations = options.pq_kmeans_iterations;
  pq_options.max_training_vectors = options.pq_training_samples;
  pq_options.threads = options.build_threads;
  pq_options.seed = options.seed;
  std::optional<ProductQuantizer> pq = ProductQuantizer::Train(
      embeddings, count, static_cast<size_t>(options.dimension),
      options.dimension, pq_options, error);
  if (!pq.has_value()) return nullptr;
  std::optional<HnswGraph> graph = HnswGraph::Create(
      capacity, options.m, options.m0, pq->code_bytes(), options.seed, error);
  if (!graph.has_value()) return nullptr;

  std::unique_ptr<ShardIndex> index(
      new ShardIndex(options, std::move(*vectors), PropertyStore(capacity),
                     std::move(*pq), std::move(*graph)));
  if (!index->AppendRows(embeddings, records, count, error)) return nullptr;
  index->builder_->InsertRange(0, static_cast<uint32_t>(count),
                               options.build_threads);
  index->RebuildFilter(count);
  index->published_.store(count, std::memory_order_release);
  return index;
}

bool ShardIndex::AppendRows(const float* embeddings,
                            const PropertyRecord* records, size_t count,
                            std::string* error) {
  if (vectors_.size() + count > vectors_.capacity()) {
    return Fail("shard capacity exceeded (" +
                    std::to_string(vectors_.capacity()) + " vectors)",
                error);
  }
  for (size_t i = 0; i < count; ++i) {
    if (!routing::IsValidLatLng(records[i].location)) {
      return Fail("record " + std::to_string(records[i].id) +
                      " has an invalid location",
                  error);
    }
  }
  const size_t dimension = static_cast<size_t>(options_.dimension);
  for (size_t i = 0; i < count; ++i) {
    const float* embedding = embeddings + i * dimension;
    const std::optional<uint32_t> row = vectors_.Append(embedding);
    properties_.Append(records[i]);
    pq_.Encode(embedding, graph_.MutableCode(*row));
  }
  return true;
}

bool ShardIndex::Insert(const float* embeddings, const PropertyRecord* records,
                        size_t count, std::string* error) {
  const size_t begin = vectors_.size();
  if (!AppendRows(embeddings, records, count, error)) return false;
  for (size_t node = begin; node < begin + count; ++node) {
    builder_->Insert(static_cast<uint32_t>(node));
  }
  RebuildFilter(begin + count);
  published_.store(begin + count, std::memory_order_release);
  return true;
}

void ShardIndex::RebuildFilter(size_t count) {
  auto filter = std::make_unique<const GeoFilter>(
      GeoFilter::Build(properties_.cell6_data(), count));
  const GeoFilter* fresh = filter.get();
  std::lock_guard<std::mutex> lock(retired_mutex_);
  const auto now = std::chrono::steady_clock::now();
  if (current_filter_ != nullptr) {
    retired_.emplace_back(now, std::move(current_filter_));
  }
  current_filter_ = std::move(filter);
  filter_.store(fresh, std::memory_order_release);
  std::erase_if(retired_, [now](const auto& entry) {
    return now - entry.first > kFilterGracePeriod;
  });
}

void ShardIndex::ExhaustiveSearch(
    const float* embedding, const SearchPredicate* predicate,
    const GeoFilter* filter, const std::vector<routing::GeohashCell>* cells,
    uint32_t published, int k, SearchScratch* scratch,
    std::vector<Neighbor>* results, uint32_t* evaluations) const {
  std::vector<Neighbor>& heap = scratch->rerank;
  heap.clear();
  const size_t limit = static_cast<size_t>(k);
  auto consider = [&](uint32_t node) {
    if (node >= published) return;
    if (predicate != nullptr && !predicate->Admits(node)) return;
    ++*evaluations;
    PushBounded(
        &heap,
        {L2SquaredDistance(embedding, vectors_.Row(node), options_.dimension),
         node},
        limit);
  };
  if (cells != nullptr && filter != nullptr) {
    for (const routing::GeohashCell& cell : *cells) {
      const IdRange range = filter->Range(cell);
      for (size_t i = range.begin; i < range.end; ++i) {
        consider(filter->ids()[i]);
      }
    }
  } else {
    for (uint32_t node = 0; node < published; ++node) consider(node);
  }
  std::sort(heap.begin(), heap.end(), CloserFirst());
  results->assign(heap.begin(), heap.end());
}

void ShardIndex::Search(const IndexQuery& query, SearchScratch* scratch,
                        std::vector<IndexHit>* hits,
                        IndexSearchStats* stats) const {
  hits->clear();
  *stats = IndexSearchStats();
  const uint32_t published = static_cast<uint32_t>(size());
  if (published == 0 || query.k <= 0 || query.embedding == nullptr) return;

  SearchParams params;
  params.k = query.k;
  params.ef = std::max(query.ef > 0 ? query.ef : options_.ef_search, query.k);
  params.rerank_depth = std::max(options_.rerank_depth, query.k);
  std::vector<Neighbor> results;
  results.reserve(static_cast<size_t>(query.k));
  uint32_t evaluations = 0;

  const bool geographic = query.radius_km > 0.0;
  if (geographic || query.sold_before_epoch_day > 0) {
    const GeoFilter* filter = nullptr;
    std::vector<routing::GeohashCell> cells;
    bool covered = false;
    if (geographic) {
      filter = filter_.load(std::memory_order_acquire);
      std::optional<routing::DiskCover> cover =
          routing::CoverDisk(query.center, query.radius_km, kPrefilterPrecision,
                             kMaxPrefilterRings);
      if (cover.has_value()) {
        cells = std::move(cover->cells);
        covered = true;
      }
    }
    const SearchPredicate predicate(&properties_, query.center, query.radius_km,
                                    cells, query.sold_before_epoch_day);
    // Without a disk cover the whole shard is the (upper-bound) population.
    const size_t population = covered ? filter->Population(cells) : published;
    stats->prefilter_population = static_cast<uint32_t>(population);
    const std::vector<routing::GeohashCell>* scan_cells =
        covered ? &cells : nullptr;
    if (query.exact || population <= options_.exhaustive_scan_threshold) {
      stats->exhaustive = true;
      ExhaustiveSearch(query.embedding, &predicate, filter, scan_cells,
                       published, query.k, scratch, &results, &evaluations);
    } else {
      // Generous budget: a few hundred beam widths of node scores.
      params.visit_budget = 256 * static_cast<size_t>(params.ef);
      SearchCounters counters;
      searcher_.Search(query.embedding, params, &predicate, published, scratch,
                       &results, &counters);
      evaluations += counters.distance_evaluations;
      if (results.size() < static_cast<size_t>(query.k) &&
          population <=
              kFallbackScanFactor * options_.exhaustive_scan_threshold) {
        stats->exhaustive = true;
        ExhaustiveSearch(query.embedding, &predicate, filter, scan_cells,
                         published, query.k, scratch, &results, &evaluations);
      }
    }
  } else if (query.exact) {
    stats->exhaustive = true;
    ExhaustiveSearch(query.embedding, nullptr, nullptr, nullptr, published,
                     query.k, scratch, &results, &evaluations);
  } else {
    SearchCounters counters;
    searcher_.Search(query.embedding, params, nullptr, published, scratch,
                     &results, &counters);
    evaluations += counters.distance_evaluations;
  }

  stats->distance_evaluations = evaluations;
  hits->reserve(results.size());
  for (const Neighbor& result : results) {
    IndexHit hit;
    hit.node = result.node;
    hit.squared_l2 = result.distance;
    hit.geo_distance_km = static_cast<float>(
        routing::HaversineKm(query.center, properties_.location(result.node)));
    hits->push_back(hit);
  }
}

size_t ShardIndex::ResidentBytes() const {
  const GeoFilter* filter = filter_.load(std::memory_order_acquire);
  return vectors_.ResidentBytes() + properties_.ResidentBytes() +
         graph_.ResidentBytes() + pq_.codebooks().size() * sizeof(float) +
         (filter != nullptr ? filter->ResidentBytes() : 0);
}

std::vector<MemoryRegion> ShardIndex::MemoryRegions() const {
  return {{vectors_.vectors_data(), vectors_.vectors_bytes()},
          {vectors_.norms_data(), vectors_.norms_bytes()},
          {graph_.arena_data(), graph_.arena_bytes()}};
}

bool ShardIndex::Save(const std::string& path, std::string* error) const {
  const size_t count = size();
  std::optional<io::SectionWriter> writer =
      io::SectionWriter::Open(path, error);
  if (!writer.has_value()) return false;

  FileHeader header{};
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.version = kFormatVersion;
  header.dimension = static_cast<uint32_t>(options_.dimension);
  header.count = count;
  header.capacity = capacity();
  header.m = options_.m;
  header.m0 = options_.m0;
  header.ef_construction = options_.ef_construction;
  header.ef_search = options_.ef_search;
  header.rerank_depth = options_.rerank_depth;
  header.pq_subspaces = options_.pq_subspaces;
  header.pq_kmeans_iterations = options_.pq_kmeans_iterations;
  header.build_threads = options_.build_threads;
  header.pq_training_samples = options_.pq_training_samples;
  header.exhaustive_scan_threshold = options_.exhaustive_scan_threshold;
  header.max_elements = options_.max_elements;
  header.seed = options_.seed;
  const HnswGraph::EntryPoint entry = graph_.entry();
  header.entry_node = entry.node;
  header.entry_level = entry.level;

  const size_t dimension = static_cast<size_t>(options_.dimension);
  std::vector<float> packed(count * dimension);
  for (size_t row = 0; row < count; ++row) {
    std::memcpy(packed.data() + row * dimension,
                vectors_.Row(static_cast<uint32_t>(row)),
                dimension * sizeof(float));
  }
  std::vector<int64_t> ids(count);
  std::vector<routing::LatLng> locations(count);
  std::vector<PropertyAttributes> attributes(count);
  for (size_t row = 0; row < count; ++row) {
    const uint32_t node = static_cast<uint32_t>(row);
    ids[row] = properties_.id(node);
    locations[row] = properties_.location(node);
    attributes[row] = properties_.attributes(node);
  }
  const std::vector<float>& codebooks = pq_.codebooks();
  const bool written =
      writer->Write(kTagHeader, &header, sizeof(header)) &&
      writer->Write(kTagVectors, packed.data(),
                    packed.size() * sizeof(float)) &&
      writer->Write(kTagCodebooks, codebooks.data(),
                    codebooks.size() * sizeof(float)) &&
      writer->Write(kTagIds, ids.data(), ids.size() * sizeof(int64_t)) &&
      writer->Write(kTagLocations, locations.data(),
                    locations.size() * sizeof(routing::LatLng)) &&
      writer->Write(kTagAttributes, attributes.data(),
                    attributes.size() * sizeof(PropertyAttributes)) &&
      writer->Write(kTagLevels, graph_.levels().data(),
                    graph_.levels().size()) &&
      writer->Write(kTagArena, graph_.arena_data(), graph_.arena_bytes());
  if (!written) {
    writer->Close(nullptr);
    return Fail("cannot write index file " + path, error);
  }
  return writer->Close(error);
}

std::unique_ptr<ShardIndex> ShardIndex::Load(const std::string& path,
                                             std::string* error) {
  std::optional<io::SectionReader> reader =
      io::SectionReader::Open(path, error);
  if (!reader.has_value()) return nullptr;
  FileHeader header{};
  if (!reader->ReadInto(kTagHeader, &header, sizeof(header), error)) {
    return nullptr;
  }
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
      header.version != kFormatVersion) {
    Fail(path + " is not a version-1 Vectra index", error);
    return nullptr;
  }
  IndexOptions options;
  options.dimension = static_cast<int>(header.dimension);
  options.m = header.m;
  options.m0 = header.m0;
  options.ef_construction = header.ef_construction;
  options.ef_search = header.ef_search;
  options.rerank_depth = header.rerank_depth;
  options.pq_subspaces = header.pq_subspaces;
  options.pq_kmeans_iterations = header.pq_kmeans_iterations;
  options.build_threads = header.build_threads;
  options.pq_training_samples = header.pq_training_samples;
  options.exhaustive_scan_threshold = header.exhaustive_scan_threshold;
  options.max_elements = header.max_elements;
  options.seed = header.seed;
  if (const std::string problem = ValidateOptions(options); !problem.empty()) {
    Fail(problem, error);
    return nullptr;
  }
  const size_t count = header.count;
  const size_t capacity = header.capacity;
  const size_t dimension = header.dimension;

  std::vector<float> packed(count * dimension);
  std::vector<float> codebooks(dimension * kPqCentroids);
  std::vector<int64_t> ids(count);
  std::vector<routing::LatLng> locations(count);
  std::vector<PropertyAttributes> attributes(count);
  std::vector<uint8_t> levels(capacity);
  std::vector<std::byte> arena;
  if (!reader->ReadInto(kTagVectors, packed.data(),
                        packed.size() * sizeof(float), error) ||
      !reader->ReadInto(kTagCodebooks, codebooks.data(),
                        codebooks.size() * sizeof(float), error) ||
      !reader->ReadInto(kTagIds, ids.data(), ids.size() * sizeof(int64_t),
                        error) ||
      !reader->ReadInto(kTagLocations, locations.data(),
                        locations.size() * sizeof(routing::LatLng), error) ||
      !reader->ReadInto(kTagAttributes, attributes.data(),
                        attributes.size() * sizeof(PropertyAttributes),
                        error) ||
      !reader->ReadInto(kTagLevels, levels.data(), levels.size(), error) ||
      !reader->ReadBytes(kTagArena, &arena, error)) {
    return nullptr;
  }
  std::optional<ProductQuantizer> pq = ProductQuantizer::FromCodebooks(
      options.dimension, options.pq_subspaces, std::move(codebooks), error);
  if (!pq.has_value()) return nullptr;
  std::optional<HnswGraph> graph = HnswGraph::FromSaved(
      capacity, options.m, options.m0, pq->code_bytes(), std::move(levels),
      arena.data(), arena.size(), error);
  if (!graph.has_value()) return nullptr;
  if (header.entry_node != HnswGraph::kNoNode) {
    graph->SetEntry(header.entry_node, header.entry_level);
  }
  std::optional<FlatStore> vectors =
      FlatStore::Create(options.dimension, capacity);
  if (!vectors.has_value()) {
    Fail("cannot allocate the vector store", error);
    return nullptr;
  }
  PropertyStore properties(capacity);
  for (size_t row = 0; row < count; ++row) {
    vectors->Append(packed.data() + row * dimension);
    PropertyRecord record;
    record.id = ids[row];
    record.location = locations[row];
    record.attributes = attributes[row];
    if (!properties.Append(record).has_value()) {
      Fail("index file holds an invalid location", error);
      return nullptr;
    }
  }
  std::unique_ptr<ShardIndex> index(
      new ShardIndex(options, std::move(*vectors), std::move(properties),
                     std::move(*pq), std::move(*graph)));
  index->RebuildFilter(count);
  index->published_.store(count, std::memory_order_release);
  return index;
}

}  // namespace vectra::index
