"""SidecarReader: read a fixed row range from the sidecar parquet.

Each SLURM array task gets a rank. The reader computes the row range
``[rank * rows_per_task, (rank+1) * rows_per_task)`` and yields
Document objects for those rows, using PyArrow row-group-level seeking
to avoid reading the entire file.
"""

from __future__ import annotations

import pyarrow.parquet as pq
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from pipeline.log import logger


class SidecarReader(PipelineStep):
    """Read a fixed row range from the sidecar parquet."""

    name = "SidecarReader"
    type = "reader"

    def __init__(
        self,
        sidecar_path: str,
        rows_per_task: int,
        columns: tuple[str, ...] = ("doc_id", "text", "safety_score"),
    ):
        super().__init__()
        self.sidecar_path = sidecar_path
        self.rows_per_task = rows_per_task
        self.columns = list(columns)

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """Yield Documents for the row range assigned to this rank.

        Uses row-group metadata to skip groups that don't overlap with
        the target range, keeping memory usage proportional to one
        row group at a time.
        """
        start = rank * self.rows_per_task
        end = start + self.rows_per_task

        pf = pq.ParquetFile(self.sidecar_path)
        total_rows = pf.metadata.num_rows

        # Clamp to actual file size
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

        # Walk row groups, skip those entirely outside our range
        row_offset = 0
        rows_yielded = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_num_rows = pf.metadata.row_group(rg_idx).num_rows
            rg_start = row_offset
            rg_end = row_offset + rg_num_rows

            # Skip row groups entirely before or after our range
            if rg_end <= start or rg_start >= end:
                row_offset = rg_end
                continue

            # Read this row group
            table = pf.read_row_group(rg_idx, columns=self.columns)

            # Slice to our range within this row group
            slice_start = max(0, start - rg_start)
            slice_end = min(rg_num_rows, end - rg_start)
            table = table.slice(slice_start, slice_end - slice_start)

            # Batch convert to Python lists (much faster than per-row .as_py())
            col_data = table.to_pydict()
            for i in range(table.num_rows):
                global_idx = rg_start + slice_start + i
                doc = Document(
                    text=col_data["text"][i],
                    id=col_data["doc_id"][i],
                    metadata={
                        "global_row_idx": global_idx,
                        "safety_score": col_data.get("safety_score", [None] * table.num_rows)[i],
                    },
                )
                self.stat_update("documents")
                yield doc
                rows_yielded += 1

            row_offset = rg_end

        logger.info("Rank {}: yielded {} documents", rank, rows_yielded)
