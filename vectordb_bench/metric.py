import logging
from dataclasses import dataclass, field

import ir_measures
import numpy as np
from ir_measures import RR, Qrel, Recall, ScoredDoc, nDCG
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for both performace and streaming cases
    insert_duration: float = 0.0
    optimize_duration: float = 0.0
    load_duration: float = 0.0  # insert + optimize

    # for performance cases
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    serial_latency_p95: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    conc_num_list: list[int] = field(default_factory=list)
    conc_qps_list: list[float] = field(default_factory=list)
    conc_latency_p99_list: list[float] = field(default_factory=list)
    conc_latency_p95_list: list[float] = field(default_factory=list)
    conc_latency_avg_list: list[float] = field(default_factory=list)

    # for streaming cases
    st_ideal_insert_duration: int = 0
    st_search_stage_list: list[int] = field(default_factory=list)
    st_search_time_list: list[float] = field(default_factory=list)
    st_max_qps_list_list: list[float] = field(default_factory=list)
    st_recall_list: list[float] = field(default_factory=list)
    st_ndcg_list: list[float] = field(default_factory=list)
    st_serial_latency_p99_list: list[float] = field(default_factory=list)
    st_serial_latency_p95_list: list[float] = field(default_factory=list)
    st_conc_failed_rate_list: list[float] = field(default_factory=list)

    # for streaming cases - concurrent latency data per stage
    st_conc_num_list_list: list[list[int]] = field(default_factory=list)
    st_conc_qps_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_p99_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_p95_list_list: list[list[float]] = field(default_factory=list)
    st_conc_latency_avg_list_list: list[list[float]] = field(default_factory=list)


class IRMetrics(BaseModel):
    """FTS metrics calculated using ir_measures library.

    Evaluates full-text search quality against human relevance judgments (qrels).
    MSMARCO qrels are sparse (~1 relevant doc per query).

    Current implementation uses a single K value. Fields are designed to support
    future multi-K evaluation (e.g., @10, @50, @100) by extending to dict[int, float].
    """

    k: int = Field(default=100, description="Cutoff value used for @K metrics evaluation.")

    recall: float = Field(
        default=0.0,
        description="Recall@K - Fraction of relevant documents (from human qrels) "
        "retrieved in top-K results. Measures coverage of relevant documents.",
    )

    ndcg: float = Field(
        default=0.0,
        description="nDCG@K - Normalized Discounted Cumulative Gain. "
        "Measures ranking quality with position-based discount. "
        "Higher positions contribute more to the score.",
    )

    mrr: float = Field(
        default=0.0,
        description="MRR@K - Mean Reciprocal Rank. "
        "Average of 1/rank for the first relevant document. "
        "Measures how quickly the first relevant result appears.",
    )


def calc_fts_metrics_ir(
    k: int,
    qrels: dict[int, list[int]],
    results: dict[int, list[int]],
) -> IRMetrics:
    """Calculate FTS metrics using ir_measures library.

    Evaluates against human relevance judgments (qrels).
    MSMARCO qrels are sparse (~1 relevant doc per query).

    Args:
        k: Cutoff for evaluation (e.g., 10, 50, 100)
        qrels: Human relevance judgments {query_id: [relevant_doc_ids]}
        results: System results to evaluate {query_id: [doc_ids in rank order]}

    Returns:
        IRMetrics containing k, recall, ndcg, mrr
    """
    # Convert system results to ScoredDoc format
    # Score = reverse rank (higher score = better rank)
    run = [
        ScoredDoc(str(qid), str(did), float(len(dids) - i))
        for qid, dids in results.items()
        for i, did in enumerate(dids)
    ]

    # Metrics from human qrels (MRR, Recall, nDCG)
    qrels_ir = [Qrel(str(qid), str(did), 1) for qid, dids in qrels.items() for did in dids]
    qrels_metrics = ir_measures.calc_aggregate([RR @ k, Recall @ k, nDCG @ k], qrels_ir, run)

    return IRMetrics(
        k=k,
        recall=qrels_metrics[Recall @ k],
        ndcg=qrels_metrics[nDCG @ k],
        mrr=qrels_metrics[RR @ k],
    )


QURIES_PER_DOLLAR_METRIC = "QP$ (Quries per Dollar)"
LOAD_DURATION_METRIC = "load_duration"
SERIAL_LATENCY_P99_METRIC = "serial_latency_p99"
SERIAL_LATENCY_P95_METRIC = "serial_latency_p95"
MAX_LOAD_COUNT_METRIC = "max_load_count"
QPS_METRIC = "qps"
RECALL_METRIC = "recall"
NDCG_METRIC = "ndcg"
MRR_METRIC = "mrr"

metric_unit_map = {
    LOAD_DURATION_METRIC: "s",
    SERIAL_LATENCY_P99_METRIC: "ms",
    SERIAL_LATENCY_P95_METRIC: "ms",
    MAX_LOAD_COUNT_METRIC: "K",
    QURIES_PER_DOLLAR_METRIC: "K",
    QPS_METRIC: "",
    RECALL_METRIC: "",
    NDCG_METRIC: "",
    MRR_METRIC: "",
}

lower_is_better_metrics = [
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
]

metric_order = [
    QPS_METRIC,
    RECALL_METRIC,
    NDCG_METRIC,
    MRR_METRIC,
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    SERIAL_LATENCY_P95_METRIC,
    MAX_LOAD_COUNT_METRIC,
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lower_is_better_metrics


def calc_recall(count: int, ground_truth: list[int], got: list[int]) -> float:
    recalls = np.zeros(count)
    for i, result in enumerate(got):
        if result in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)


def get_ideal_dcg(k: int):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)

    return ideal_dcg


def calc_ndcg(ground_truth: list[int], got: list[int], ideal_dcg: float) -> float:
    dcg = 0
    ground_truth = list(ground_truth)
    for got_id in set(got):
        if got_id in ground_truth:
            idx = ground_truth.index(got_id)
            dcg += 1 / np.log2(idx + 2)
    return dcg / ideal_dcg


def calc_mrr(ground_truth: list[int], got: list[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR is the average of the reciprocal ranks of the first relevant result
    for each query. If no relevant result is found, MRR is 0.

    Args:
        ground_truth: List of relevant document IDs
        got: List of retrieved document IDs (in order)

    Returns:
        MRR score (0-1)
    """
    ground_truth_set = set(ground_truth)
    for rank, doc_id in enumerate(got, start=1):
        if doc_id in ground_truth_set:
            return 1.0 / rank
    return 0.0


def calc_recall_fts(k: int, ground_truth: list[int], got: list[int]) -> float:
    if not ground_truth or k == 0:
        return 0.0
    gt_set = set(ground_truth)
    retrieved_top_k = set(got[:k])
    if not gt_set:
        return 0.0
    return len(gt_set & retrieved_top_k) / len(gt_set)


def calc_ndcg_fts(k: int, ground_truth: list[int], got: list[int]) -> float:
    if not ground_truth or k == 0:
        return 0.0
    ground_truth_set = set(ground_truth)
    dcg = 0.0
    for position, doc_id in enumerate(got[:k]):
        if doc_id in ground_truth_set:
            dcg += 1.0 / np.log2(position + 2)
    num_relevant = len(ground_truth_set)
    idcg = 0.0
    limit = min(k, num_relevant)
    for i in range(limit):
        idcg += 1.0 / np.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
