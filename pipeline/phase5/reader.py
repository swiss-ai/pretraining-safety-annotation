"""PromptsReader: read a fixed row range from prompts.parquet.

Each SLURM array task gets a rank. The reader computes the row range
``[rank * rows_per_task, (rank+1) * rows_per_task)`` and yields
Document objects for those rows. Mirrors ``pipeline/phase4/reader.py``.
"""
from __future__ import annotations

import pyarrow.parquet as pq
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from pipeline.log import logger


class PromptsReader(PipelineStep):
    """Read a fixed row range from the materialised prompts parquet."""

    name = "PromptsReader"
    type = "reader"

    def __init__(
        self,
        prompts_path: str,
        rows_per_task: int,
    ):
        super().__init__()
        self.prompts_path = prompts_path
        self.rows_per_task = rows_per_task

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """Yield Documents for the row range assigned to this rank."""
        start = rank * self.rows_per_task
        end = start + self.rows_per_task

        pf = pq.ParquetFile(self.prompts_path)
        total_rows = pf.metadata.num_rows

        if start >= total_rows:
            logger.info(
                "Rank {} has no rows (start={} >= total={})", rank, start, total_rows
            )
            return
        end = min(end, total_rows)

        logger.info(
            "Rank {}: reading rows [{}, {}) of {} total",
            rank, start, end, total_rows,
        )

        row_offset = 0
        rows_yielded = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_num_rows = pf.metadata.row_group(rg_idx).num_rows
            rg_start = row_offset
            rg_end = row_offset + rg_num_rows

            if rg_end <= start or rg_start >= end:
                row_offset = rg_end
                continue

            table = pf.read_row_group(rg_idx)
            slice_start = max(0, start - rg_start)
            slice_end = min(rg_num_rows, end - rg_start)
            table = table.slice(slice_start, slice_end - slice_start)

            col_data = table.to_pydict()
            for i in range(table.num_rows):
                # Trust the stored column rather than recompute from row
                # position — survives any future re-sort of prompts.parquet.
                doc = Document(
                    text=col_data["user"][i],
                    id=col_data["source_id"][i],
                    metadata={
                        "global_row_idx": col_data["global_row_idx"][i],
                        "source": col_data["source"][i],
                        "source_id": col_data["source_id"][i],
                        "user": col_data["user"][i],
                        "meta": col_data["meta"][i],
                        "harm_category": col_data["harm_category"][i],
                    },
                )
                self.stat_update("documents")
                yield doc
                rows_yielded += 1

            row_offset = rg_end

        logger.info("Rank {}: yielded {} documents", rank, rows_yielded)
