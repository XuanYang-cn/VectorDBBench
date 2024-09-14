import logging
from typing import Iterable
import multiprocessing as mp

from .mp_runner import MultiProcessingSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench import config
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager

log = logging.getLogger(__name__)


class ReadWriteRunner(MultiProcessingSearchRunner, RatedMultiThreadingInsertRunner):
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        insert_rate: int = 10,
        normalize: bool = False,
        k: int = 100,
        filters: dict | None = None,
        concurrencies: Iterable[int] = (1, 15),
        duration: int = 30,
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size

        import numpy as np
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
        import concurrent
        futures = []
        with mp.Manager() as m:
            q = m.Queue(maxsize=100)
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                futures.append(executor.submit(self.run_with_rate, q))
                futures.append(executor.submit(self.run_search_by_sig, q))

                for future in concurrent.futures.as_completed(futures):
                    future.result()

        log.info("Concurrent read write all done")


    def run_search_by_sig(self, q):
        # run search half way
        insert_count = 0
        while q.get(block=True) is not None:
            insert_count += config.NUM_PER_BATCH * self.insert_rate
            if insert_count >= self.data_volume // 2:
                log.info("Insert halfway done, run concurrent search")
                self.run()
                break

        while q.get(block=True) is not None:
            pass

        log.info("Insert done, run the second concurrent search")
        # run search when insertion done.
        self.run()
