import logging
from typing import Iterable
import multiprocessing as mp
import concurrent
import numpy as np

from .mp_runner import MultiProcessingSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager

log = logging.getLogger(__name__)


class ReadWriteRunner(MultiProcessingSearchRunner, RatedMultiThreadingInsertRunner):
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        insert_rate: int = 1000,
        normalize: bool = False,
        k: int = 100,
        filters: dict | None = None,
        concurrencies: Iterable[int] = (1, 15, 50),
        duration: int = 120, # TODO, choose duration based on dur of insert 10% data
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size

        test_emb = np.stack(dataset.test_data["emb"])
        if normalize:
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
        test_emb = test_emb.tolist()

        MultiProcessingSearchRunner.__init__(
            self, db, test_emb, k, filters, concurrencies, duration
        )
        RatedMultiThreadingInsertRunner.__init__(
            self,
            rate=insert_rate,
            db=db,
            dataset_iter=iter(dataset),
            normalize=normalize,
        )

    def run_read_write(self):
        futures = []
        with mp.Manager() as m:
            q = m.Queue()
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                futures.append(executor.submit(self.run_with_rate, q))
                futures.append(executor.submit(self.run_search_by_sig, q))

                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    log.Info(f"Result = {res}")

        log.info("Concurrent read write all done")


    def run_search_by_sig(self, q):
        res = []
        total_batch = self.data_volume // self.insert_rate
        batch = 1
        while q.get(block=True) is not None:
            perc = batch * 100 / total_batch
            if perc % 10 == 0:
                log.info(f"Insert {perc}% done, total batch={total_batch}")
                if perc / 10 >= 5:
                    log.info(f"Insert {perc}% done, run concurrent search")
                    max_qps, _, _, _ = self.run()
                    res.append((perc, max_qps))

            batch += 1
        return res
