from vectordb_bench.backend.dataset import Dataset, DatasetSource
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.config import FLATConfig
import logging

log = logging.getLogger("vectordb_bench")
log.setLevel(logging.DEBUG)

db_config = {
    "uri": "http://127.0.0.1:19530",
}

def get_rate_runner(db):
    cohere = Dataset.COHERE.manager(100_000)
    prepared = cohere.prepare(DatasetSource.AliyunOSS)
    assert prepared
    runner = RatedMultiThreadingInsertRunner(
        rate = 10,
        db = db,
        dataset = cohere,
    )

    return runner

def test_rate_runner_milvus():
    milvus = DB.Milvus.init_cls(dim=768, db_config=db_config, db_case_config=FLATConfig(metric_type="COSINE"), drop_old=True)
    runner = get_rate_runner(milvus)

    _, t = runner.run_with_rate()
    log.info(f"insert run done, time={t}")


if __name__ == "__main__":
    test_rate_runner_milvus()

    from pymilvus import MilvusClient
    c = MilvusClient(**db_config)

    coll = c.list_collections()[0]
    log.info(f"Coll: {coll}")
    c._get_connection().flush([coll])
    log.info("Coll flushed")
    c.load_collection(coll)
    log.info("Coll loaded")
    ret = c.query(coll, output_fields=["count(*)"], consistency_level="Strong")
    log.info(f"Count, {ret}")
