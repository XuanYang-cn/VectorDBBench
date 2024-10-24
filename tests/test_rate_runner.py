import argparse
from vectordb_bench.backend.dataset import Dataset, DatasetSource
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner
from vectordb_bench.backend.clients import DB, VectorDB
from vectordb_bench.backend.clients.milvus.config import FLATConfig
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig

import logging

log = logging.getLogger("vectordb_bench")
log.setLevel(logging.DEBUG)

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

def test_rate_runner(db, insert_rate):
    runner = get_rate_runner(db)

    _, t = runner.run_with_rate()
    log.info(f"insert run done, time={t}")

def test_read_write_runner(db, insert_rate, conc: list, local: bool=False):
    cohere = Dataset.COHERE.manager(10_000_000)
    if local is True:
        source = DatasetSource.AliyunOSS
    else:
        source = DatasetSource.S3
    prepared = cohere.prepare(source)
    assert prepared

    rw_runner = ReadWriteRunner(
        db=db,
        dataset=cohere,
        insert_rate=insert_rate,
        concurrencies=conc
    )
    rw_runner.run_read_write()


def get_db(db: str, config: dict) -> VectorDB:
    if db == DB.Milvus.name:
        return DB.Milvus.init_cls(dim=768, db_config=config, db_case_config=FLATConfig(metric_type="COSINE"), drop_old=True, pre_load=True)
    elif db == DB.ZillizCloud.name:
        return DB.ZillizCloud.init_cls(dim=768, db_config=config, db_case_config=AutoIndexConfig(metric_type="COSINE"), drop_old=True, pre_load=True)
    else:
        raise ValueError(f"unknown db: {db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--insert_rate", type=int, default="1000", help="insert entity row count per seconds, cps")
    parser.add_argument("-d", "--db", type=str, default=DB.Milvus.name, help="db name")
    parser.add_argument("-c", "--conc", type=list, default=(1, 15, 50), help="search conc")
    parser.add_argument("--use_s3", action='store_true', help="whether to use S3 dataset")

    flags = parser.parse_args()

    # TODO read uri, user, password from .env
    config = {
            "uri": "http://localhost:19530",
        "user": "", 
        "password": "",
    }

    db = get_db(flags.db, config)
    test_read_write_runner(db, flags.insert_rate, flags.conc, flags.use_s3)
