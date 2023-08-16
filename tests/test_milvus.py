import logging
from vectordb_bench.models import (
    DB,
    IndexType,
    TaskConfig,
    CaseConfig,
    CaseType,
    Metric,
)

from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.backend.cases import Case

log = logging.getLogger(__name__)

_URI = "http://localhost:19530"

CASE = CaseType.Performance768D1M.case_cls()

class TTLCaseRunner(CaseRunner):
    run_id: str = "milvus_ttl"
    config: TaskConfig = TaskConfig(
        db=DB.Milvus,
        db_config=DB.Milvus.init_cls.config_cls()(
            uri=_URI,
            db_label="local_milvus"
        ),
        db_case_config=DB.Milvus.init_cls.case_config_cls(IndexType.HNSW)(
            M=30,
            efConstruction=360,
            ef=100,
            metric_type=CASE.dataset.data.metric_type,
        ),
        case_config= CaseConfig(
            case_id=CASE.case_id,
        ),
    )
    ca: Case = CASE
    status: RunningStatus = RunningStatus.PENDING

    def _run_perf_case(self, drop_old: bool = True) -> Metric:
        """Overwrite of the perfcases runner"""
        try:
            m = Metric()

            # set ttl before insertion
            tmp_db = self.config.db.init_cls(
                dim=self.ca.dataset.data.dim,
                db_config=self.config.db_config.to_dict(),
                db_case_config=self.config.db_case_config,
                drop_old=drop_old,
            )
            with tmp_db.init():
                tmp_db.col.set_properties(properties={"collection.ttl.seconds": 5 * 60 * 60}) # 5hrs

            if drop_old:
                _, load_dur = self._load_train_data()
                build_dur = self._optimize()
                m.load_duration = round(load_dur+build_dur, 4)
                log.info(
                    f"Finish loading the entire dataset into VectorDB,"
                    f" insert_duration={load_dur}, optimize_duration={build_dur}"
                    f" load_duration(insert + optimize) = {m.load_duration}"
                )

        except Exception as e:
            log.warning(f"Failed to run performance case, reason = {e}")
            raise e from None
        else:
            log.info(f"Performance case got result: {m}")
            return m


class TestMilvusTTL:
    def test_ttl(self):
        ttl_runner = TTLCaseRunner()
        ttl_runner.display()
        ttl_runner.run()
