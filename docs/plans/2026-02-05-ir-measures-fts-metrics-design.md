# Design: Integrate ir_measures for FTS Metrics

**Date:** 2026-02-05
**Status:** Implemented
**Scope:** FTS (Full-Text Search) cases only; vector search unchanged
**Author:** @XuanYang-cn

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ir_datasets (msmarco-passage/dev/small)                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│  │ queries_iter()   │  │ qrels_iter()     │  │ scoreddocs_iter()│               │
│  │ (query_id, text) │  │ (human labels)   │  │ (BM25 baseline)  │               │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘               │
│           │                     │                     │                         │
│           ▼                     ▼                     ▼                         │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │              MSMARCODatasetTranslator  [MODIFIED]              │             │
│  │                                                                │             │
│  │  load_test_data() [NEW] - handles all filtering internally     │             │
│  │  ┌─────────────────────────────────────────────────────────┐   │             │
│  │  │ 1. Build qrels → get evaluable_query_ids                │   │             │
│  │  │ 2. Filter queries by evaluable_query_ids                │   │             │
│  │  │ 3. Filter scoredocs by evaluable_query_ids (if avail)   │   │             │
│  │  │                                                         │   │             │
│  │  │ Returns: (list[FtsQuery], FtsGroundTruthData)           │   │             │
│  │  │          Only evaluable queries + their ground truth    │   │             │
│  │  └─────────────────────────────────────────────────────────┘   │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                      │                                          │
│                                      ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │                    FtsDatasetManager  [MODIFIED]               │             │
│  │                                                                │             │
│  │  queries_data, gt_data = translator.load_test_data(...)     │             │
│  │                                                                │             │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────┐  │             │
│  │  │ queries_data            │  │ gt_data                  │  │             │
│  │  │ list[FtsQuery]          │  │ FtsGroundTruthData          │  │             │
│  │  │ (evaluable only)        │  │ (matching queries only)     │  │             │
│  │  └─────────────────────────┘  └─────────────────────────────┘  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTION LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │                   TaskRunner  [MODIFIED]                       │             │
│  │                                                                │             │
│  │  _init_fts_search_runner():                                    │             │
│  │    - Pass fts_dataset.queries_data (already filtered)          │             │
│  │    - Pass fts_dataset.gt_data to runner                     │             │
│  │                                                                │             │
│  └──────────────────────────┬─────────────────────────────────────┘             │
│                             │                                                    │
│                             ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │               SerialSearchRunner  [MODIFIED]                   │             │
│  │                                                                │             │
│  │  NEW param: fts_ground_truth: FtsGroundTruthData | None        │             │
│  │                                                                │             │
│  │  ┌─────────────────────────────────────────────────────────┐   │             │
│  │  │  FTS Path (when fts_ground_truth provided):             │   │             │
│  │  │                                                         │   │             │
│  │  │  for query in test_data:  # list[FtsQuery]              │   │             │
│  │  │      results = search(query.text)                       │   │             │
│  │  │      results_dict[query.query_id] = results             │   │             │
│  │  │                                                         │   │             │
│  │  │  ir_metrics = calc_fts_metrics_ir(...)  ──────────────────────────┐       │
│  │  │  return (recall, ndcg, mrr, math_recall, p99, p95)      │   │     │       │
│  │  └─────────────────────────────────────────────────────────┘   │     │       │
│  │                                                                │     │       │
│  │  ┌─────────────────────────────────────────────────────────┐   │     │       │
│  │  │  Vector Path (unchanged):                               │   │     │       │
│  │  │  - Per-query calc_recall(), calc_ndcg(), calc_mrr()     │   │     │       │
│  │  │  - return (recall, ndcg, mrr, p99, p95)                 │   │     │       │
│  │  └─────────────────────────────────────────────────────────┘   │     │       │
│  └────────────────────────────────────────────────────────────────┘     │       │
│                                                                         │       │
└─────────────────────────────────────────────────────────────────────────┼───────┘
                                                                          │
                                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            METRICS LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  metric.py  [MODIFIED]                                                          │
│                                                                                  │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │  IRMetrics [NEW]                │  │  calc_fts_metrics_ir() [NEW]        │   │
│  │  (Pydantic BaseModel)           │  │                                     │   │
│  │                                 │  │  Uses ir_measures library:          │   │
│  │  - k: int                       │  │  - RR@k   → mrr                     │   │
│  │  - recall: float                │  │  - Recall@k → recall                │   │
│  │  - ndcg: float                  │  │  - nDCG@k → ndcg                    │   │
│  │  - mrr: float                   │  │  - Recall@k (vs BM25) → math_recall │   │
│  │  - math_recall: float | None    │  │                                     │   │
│  │                                 │  │  Returns: IRMetrics                 │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘   │
│                                                                                  │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │  Metric [MODIFIED]              │  │  Existing (unchanged):              │   │
│  │  (dataclass)                    │  │  - calc_recall()                    │   │
│  │                                 │  │  - calc_ndcg()                      │   │
│  │  + math_recall: float | None    │  │  - calc_mrr()                       │   │
│  │    (NEW field)                  │  │  - calc_recall_fts()                │   │
│  │                                 │  │  - calc_ndcg_fts()                  │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Summary

Replace custom FTS metric calculations with the `ir_measures` library and add `math_recall` metric. This brings standardized IR evaluation metrics to VectorDBBench FTS benchmarks while maintaining backwards compatibility for vector search cases.

## Motivation

- **Standardization:** `ir_measures` is the standard library for IR evaluation, used widely in research
- **Correctness:** Library-based calculations are better tested than custom implementations
- **New metric:** `math_recall` measures overlap with BM25 baseline, useful for comparing dense vs sparse retrieval

## Design

### 1. Add Dependencies

**File:** `pyproject.toml`

Add `ir_datasets` and `ir_measures` to direct dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "ir_datasets",
    "ir_measures",
]
```

---

### 2. Update Data Model

**File:** `vectordb_bench/backend/dataset.py`

#### 2.1 New Ground Truth Data Structure

```python
@dataclass
class FtsGroundTruthData:
    """Ground truth data for FTS evaluation.

    Contains two distinct sources for comprehensive evaluation:
    - qrels: Human relevance judgments (sparse, ~1 relevant doc per query)
    - scoredocs: BM25 baseline rankings (optional, for math_recall metric)

    Note: scoredocs is optional because not all ir_datasets provide BM25 baselines.
    When scoredocs is None, math_recall will not be calculated.
    """
    qrels: dict[int, list[int]]                    # Human judgments -> MRR, Recall, nDCG
    scoredocs: dict[int, list[int]] | None = None  # BM25 baseline -> math_recall (optional)
```

#### 2.2 Update MSMARCODatasetTranslator

```python
def load_test_data(self, dataset: typing.Any) -> tuple[list[FtsQuery], FtsGroundTruthData]:
    """Load queries and ground truth together, returning only evaluable queries.

    The translator handles all filtering internally to avoid loading large datasets
    into memory. Only queries that have qrels (can be evaluated) are returned.

    Returns:
        tuple of:
        - list[FtsQuery]: Queries that have ground truth (can be evaluated)
        - FtsGroundTruthData: Ground truth for those queries only
    """
    # Step 1: Build qrels first to know which queries are evaluable
    qrels: dict[int, list[int]] = {}
    for qrel in dataset.qrels_iter():
        query_id = int(qrel.query_id)
        doc_id = int(qrel.doc_id)
        if qrel.relevance > 0:
            if query_id not in qrels:
                qrels[query_id] = []
            qrels[query_id].append(doc_id)

    evaluable_query_ids = set(qrels.keys())

    # Step 2: Load only queries that have qrels
    queries: list[FtsQuery] = []
    for q in dataset.queries_iter():
        query_id = int(q.query_id)
        if query_id in evaluable_query_ids:
            queries.append(FtsQuery(query_id=query_id, text=q.text))

    # Step 3: Load scoredocs only for evaluable queries (optional)
    scoredocs: dict[int, list[int]] | None = None
    if hasattr(dataset, 'scoreddocs_iter'):
        scoredocs = {}
        for sd in dataset.scoreddocs_iter():
            query_id = int(sd.query_id)
            if query_id not in evaluable_query_ids:
                continue
            doc_id = int(sd.doc_id)
            if query_id not in scoredocs:
                scoredocs[query_id] = []
            scoredocs[query_id].append(doc_id)

    return queries, FtsGroundTruthData(qrels=qrels, scoredocs=scoredocs)
```

#### 2.3 Update FtsDatasetManager

Change `gt_data` type from `FtsGroundTruth` to `FtsGroundTruthData`.

Simplified loading - translator handles all filtering:

```python
def prepare(self, ...):
    # Translator returns only evaluable queries + their ground truth
    self.queries_data, self.gt_data = self._translator.load_test_data(self._ir_dataset)
    log.info(f"Loaded {len(self.queries_data)} evaluable queries with ground truth")
```

---

### 3. Add IRMetrics Class and Calculator

**File:** `vectordb_bench/metric.py`

#### 3.1 IRMetrics Pydantic Model

```python
from pydantic import BaseModel, Field


class IRMetrics(BaseModel):
    """FTS metrics calculated using ir_measures library.

    These metrics evaluate full-text search quality against two ground truth sources:
    - Human relevance judgments (qrels): For MRR, Recall, nDCG
    - BM25 baseline results (scoredocs): For math_recall

    Current implementation uses a single K value. Fields are designed to support
    future multi-K evaluation (e.g., @10, @50, @100) by extending to dict[int, float].
    """

    # K value(s) used for evaluation
    k: int = Field(
        default=100,
        description="Cutoff value used for @K metrics evaluation."
    )
    # Future: k_values: list[int] = Field(default_factory=list)

    recall: float = Field(
        default=0.0,
        description="Recall@K - Fraction of relevant documents (from human qrels) "
                    "retrieved in top-K results. Measures coverage of relevant documents."
    )
    # Future multi-K: recall_at_k: dict[int, float] = Field(default_factory=dict)

    ndcg: float = Field(
        default=0.0,
        description="nDCG@K - Normalized Discounted Cumulative Gain. "
                    "Measures ranking quality with position-based discount. "
                    "Higher positions contribute more to the score."
    )
    # Future multi-K: ndcg_at_k: dict[int, float] = Field(default_factory=dict)

    mrr: float = Field(
        default=0.0,
        description="MRR@K - Mean Reciprocal Rank. "
                    "Average of 1/rank for the first relevant document. "
                    "Measures how quickly the first relevant result appears."
    )
    # Future multi-K: mrr_at_k: dict[int, float] = Field(default_factory=dict)

    math_recall: float | None = Field(
        default=None,
        description="Mathematical Recall@K - Overlap between system's top-K results "
                    "and BM25 baseline's top-K results. Measures similarity to "
                    "traditional BM25 retrieval, useful for comparing dense vs sparse retrieval. "
                    "None if dataset does not provide BM25 scoredocs."
    )
    # Future multi-K: math_recall_at_k: dict[int, float] = Field(default_factory=dict)
```

#### 3.2 Calculator Function

```python
def calc_fts_metrics_ir(
    k: int,
    qrels: dict[int, list[int]],
    results: dict[int, list[int]],
    scoredocs: dict[int, list[int]] | None = None,
) -> IRMetrics:
    """Calculate FTS metrics using ir_measures library.

    Uses two distinct ground truth sources for evaluation:
    1. Human qrels: Sparse relevance judgments (~1 relevant doc per query)
       Used for: MRR, Recall, nDCG
    2. BM25 scoredocs (optional): Top-K documents from BM25 baseline
       Used for: math_recall (measures overlap with traditional retrieval)

    Args:
        k: Cutoff for evaluation (e.g., 10, 50, 100)
           Future: k_values: list[int] for multi-K evaluation
        qrels: Human relevance judgments {query_id: [relevant_doc_ids]}
        results: System results to evaluate {query_id: [doc_ids in rank order]}
        scoredocs: BM25 baseline rankings {query_id: [doc_ids by BM25 score]} (optional)

    Returns:
        IRMetrics containing k, recall, ndcg, mrr, math_recall (None if no scoredocs)
    """
    from ir_measures import Recall, RR, nDCG, ScoredDoc, Qrel
    import ir_measures

    # Convert system results to ScoredDoc format
    # Score = reverse rank (higher score = better rank)
    run = [
        ScoredDoc(str(qid), str(did), float(len(dids) - i))
        for qid, dids in results.items()
        for i, did in enumerate(dids)
    ]

    # Metrics from human qrels (MRR, Recall, nDCG)
    qrels_ir = [
        Qrel(str(qid), str(did), 1)
        for qid, dids in qrels.items()
        for did in dids
    ]
    qrels_metrics = ir_measures.calc_aggregate(
        [RR@k, Recall@k, nDCG@k],
        qrels_ir,
        run
    )

    # math_recall from BM25 scoredocs (optional)
    math_recall = None
    if scoredocs is not None:
        bm25_qrels = [
            Qrel(str(qid), str(did), 1)
            for qid, dids in scoredocs.items()
            for did in dids[:k]
        ]
        math_metrics = ir_measures.calc_aggregate(
            [Recall@k],
            bm25_qrels,
            run
        )
        math_recall = math_metrics[Recall@k]

    return IRMetrics(
        k=k,
        recall=qrels_metrics[Recall@k],
        ndcg=qrels_metrics[nDCG@k],
        mrr=qrels_metrics[RR@k],
        math_recall=math_recall,
    )


# Future: Multi-K evaluation function
# def calc_fts_metrics_ir_multi_k(
#     k_values: list[int],
#     qrels: dict[int, list[int]],
#     scoredocs: dict[int, list[int]],
#     results: dict[int, list[int]],
# ) -> IRMetrics:
#     """Calculate FTS metrics at multiple K cutoffs."""
#     ...
```

#### 3.3 Update Metric Constants

```python
MATH_RECALL_METRIC = "math_recall"

metric_unit_map = {
    # ... existing entries ...
    MATH_RECALL_METRIC: "",
}

metric_order = [
    QPS_METRIC,
    RECALL_METRIC,
    NDCG_METRIC,
    MRR_METRIC,
    MATH_RECALL_METRIC,  # Add after MRR
    LOAD_DURATION_METRIC,
    # ... rest ...
]
```

---

### 4. Update Metric Dataclass

**File:** `vectordb_bench/metric.py`

Add `math_recall` field to existing `Metric` dataclass:

```python
@dataclass
class Metric:
    """result metrics"""

    # ... existing fields ...

    # for performance cases
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    serial_latency_p95: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    math_recall: float | None = None  # NEW: BM25 baseline overlap (FTS only, None if unavailable)

    # ... rest of fields ...
```

---

### 5. Update SerialSearchRunner for FTS

**File:** `vectordb_bench/backend/runner/serial_runner.py`

#### 5.1 Add Ground Truth Parameters

```python
class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list,
        ground_truth: list[list[int]],
        k: int = 100,
        filters: Filter = non_filter,
        search_fulltext: bool | None = None,
        # NEW: Optional FTS ground truth for ir_measures
        fts_ground_truth: FtsGroundTruthData | None = None,
    ):
        # ... existing init ...
        self._fts_ground_truth = fts_ground_truth
```

Note: For FTS mode, `test_data` is `list[FtsQuery]` (not plain strings). The `FtsQuery` objects contain both `query_id` and `text`, eliminating the need for separate ID tracking.

#### 5.2 Update FTS Search Method

When `_use_fts_metrics=True` and `_fts_ground_truth` is provided:
- Iterate over `FtsQuery` objects (contain both `query_id` and `text`)
- Collect results as `{query_id: [doc_ids]}`
- Call `calc_fts_metrics_ir()` once after all queries
- Return `math_recall` as additional value

```python
def search(self, args) -> tuple[float, float, float, float, float, float]:
    # ... existing setup ...

    if self._use_fts_metrics and self._fts_ground_truth is not None:
        # Batch FTS metrics using ir_measures
        # test_data is list[FtsQuery] with query_id and text
        results_dict: dict[int, list[int]] = {}
        latencies = []

        for query in test_data:  # FtsQuery objects
            s = time.perf_counter()
            results = self._get_db_search_res(query.text)
            latencies.append(time.perf_counter() - s)
            results_dict[query.query_id] = results

        # Calculate metrics using ir_measures
        ir_metrics = calc_fts_metrics_ir(
            k=self.k,
            qrels=self._fts_ground_truth.qrels,
            results=results_dict,
            scoredocs=self._fts_ground_truth.scoredocs,
        )

        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)

        return (ir_metrics.recall, ir_metrics.ndcg, ir_metrics.mrr,
                ir_metrics.math_recall, p99, p95)
    else:
        # Existing per-query calculation for vector search
        # ... unchanged ...
```

#### 5.3 Update Return Signature

```python
# Vector search (unchanged): (recall, ndcg, mrr, p99, p95)
# FTS with ir_measures:      (recall, ndcg, mrr, math_recall, p99, p95)
```

---

### 6. Update Task Runner

**File:** `vectordb_bench/backend/task_runner.py`

#### 6.1 Update _init_fts_search_runner

```python
def _init_fts_search_runner(self):
    fts_dataset = self.ca.dataset

    # queries_data already contains only evaluable queries (filtered by translator)
    log.info(f"FTS test will use {len(fts_dataset.queries_data)} queries for testing")

    self.serial_search_runner = SerialSearchRunner(
        db=self.db,
        test_data=fts_dataset.queries_data,  # list[FtsQuery] - already filtered
        ground_truth=None,  # Not used for FTS with ir_measures
        filters=self.ca.filters,
        k=self.config.case_config.k,
        search_fulltext=True,
        fts_ground_truth=fts_dataset.gt_data,  # FtsGroundTruthData
    )
```

#### 6.2 Handle math_recall in Results

```python
def _serial_search(self) -> tuple[float, float, float, float, float, float] | None:
    result, _ = self.serial_search_runner.run()
    # FTS returns 6 values, vector returns 5
    if len(result) == 6:
        recall, ndcg, mrr, math_recall, p99, p95 = result
        return recall, ndcg, mrr, math_recall, p99, p95
    else:
        recall, ndcg, mrr, p99, p95 = result
        return recall, ndcg, mrr, 0.0, p99, p95  # math_recall=0 for vector
```

#### 6.3 Populate Metric

```python
metric = Metric(
    # ... existing fields ...
    recall=recall,
    ndcg=ndcg,
    mrr=mrr,
    math_recall=math_recall,  # NEW
    serial_latency_p99=p99,
    serial_latency_p95=p95,
)
```

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Add ir_datasets, ir_measures to direct dependencies |
| `vectordb_bench/metric.py` | Add IRMetrics class, calc_fts_metrics_ir(), math_recall to Metric |
| `vectordb_bench/backend/dataset.py` | FtsGroundTruthData, load both qrels + scoredocs |
| `vectordb_bench/backend/runner/serial_runner.py` | Batch FTS metrics with ir_measures |
| `vectordb_bench/backend/task_runner.py` | Pass ground truth data, handle math_recall |

---

## Not Changed (Vector Search)

- `calc_recall()`, `calc_ndcg()`, `calc_mrr()` - unchanged
- Vector search path in `SerialSearchRunner` - unchanged
- All non-FTS cases - unchanged

---

## Future Enhancements

1. **Multi-K evaluation:** Add `k_values: list[int]` parameter and `*_at_k: dict[int, float]` fields
2. **Additional metrics:** P@K (Precision), MAP (Mean Average Precision)
3. **Per-query metrics:** Return detailed per-query scores for analysis

---

## Testing

1. Run existing FTS benchmark to verify metrics are calculated
2. Compare ir_measures results with custom implementation for validation
3. Verify math_recall values are reasonable (0-1 range, correlated with recall)
