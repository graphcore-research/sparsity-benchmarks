# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Tuple
from ._memory import Memory
from ._cycles import Cycles
from ._io import IO
import pva
from . import _utils


class Reptil:
    """
    Utility class for performing high level functions on a pva.Report
    object.
    """

    def __init__(
        self, report: pva.Report, engine_run_range: Tuple[int, int] = None
    ) -> None:
        """
        Utility class for performing high level functions on a
        pva.Report object.

        Parameters
        ----------
        report : pva.Report
            Report object that will be used by the reptil methods
        engine_run_range : Tuple[int,int]
            The range of engine runs to be used for analysis, [start, end), by
            default [0, len(runs))

        Raises
        ------
        ValueError
            If report isn't a pva.Report
        """
        if isinstance(report, pva.Report):
            self._report = report
        else:
            raise ValueError("report must be a pva.Report object.")

        if engine_run_range is not None:
            start, end = engine_run_range
            if (
                not len(range(start, end))
                or start < 0
                or end >= len(self._report.execution.runs)
            ):
                raise IndexError("Invalid run range provided")

        self._engine_run_range = engine_run_range

        self._memory: Memory = Memory(self)
        self._cycles: Cycles = Cycles(self)
        self._io: IO = IO(self)

    def _steps_generator(self):
        if self._engine_run_range is None:
            for step in self._report.execution.steps:
                yield step
        else:
            start, end = self._engine_run_range
            for i in range(start, end):
                run = self._report.execution.runs[i]
                for step in run.steps:
                    yield step

    @property
    def _steps(self):
        return self._steps_generator

    @property
    def memory(self) -> Memory:
        """
        Namespace for methods related to memory.

        Returns
        -------
        Memory
            Object that owns the memory methods.
        """
        return self._memory

    @property
    def cycles(self) -> Cycles:
        """
        Namespace for methods related to execution cycle metrics.

        Returns
        -------
        Cycles
            Object that owns the execution cycle metrics methods.
        """
        return self._cycles

    @property
    def io(self) -> IO:
        """
        Namespace for methods related to IO metrics.

        Returns
        -------
        IO
            Object that owns the IO metric methods.
        """
        return self._io

    @property
    def summary(self) -> str:
        """
        Returns a string summary of the first run in a profile report.

        Returns
        -------
        str
            Printable string giving a summary of the rest of the reptil
            methods.
        """
        run = self._report.execution.runs[0]
        total_cycles = sum(self.cycles.program().values())
        program_summary = self.cycles.program()
        total_runtime = (
            run.microseconds.end - run.microseconds.start
        ) / 1000000

        on_tile_execute = (
            program_summary.get("OnTileExecute", 0) / total_cycles
        )
        do_exchange = program_summary.get("DoExchange", 0) / total_cycles
        sync = program_summary.get("Sync", 0) / total_cycles
        sync_ans = program_summary.get("SyncAns", 0) / total_cycles
        global_exchange = (
            program_summary.get("GlobalExchange", 0) / total_cycles
        )

        on_chip_compute_percentage = (
            on_tile_execute + do_exchange + sync + sync_ans
        )

        host_io_percentage = (
            program_summary.get("StreamCopyBegin", 0)
            + program_summary.get("StreamCopyMid", 0)
            + program_summary.get("StreamCopyEnd", 0)
        ) / total_cycles

        summary_text = (
            "\nTotal running time: "
            f" {'{:.4f}'.format(total_runtime)} seconds\n\nOn-chip compute"
            f" time:  {on_chip_compute_percentage * 100:.2f} %"
            f" ({on_chip_compute_percentage * total_runtime:.4f} secs)\n    -"
            " Compute efficiency:  ~\n    - Compute / Exchange / Sync: "
            f" {on_tile_execute*100:.2f}% / {do_exchange*100:.2f}% /"
            f" {sync*100 + sync_ans*100:.2f}%\n\nInter-IPU comms time: "
            f" {global_exchange*100:.2f}%"
            f" ({global_exchange * total_runtime:.4f} secs)\n    - Average"
            " per-IPU link speed (bidirectional):  ~\n\nHost I/O time: "
            f" {host_io_percentage*100:.2f}%"
            f" ({host_io_percentage * total_runtime:.4f} secs)\n    - Average"
            " host bandwidth (bidirectional): "
            f" {self.io.bandwidth.host / 10**9:.2f} GB/s "
        )
        return summary_text

    @property
    def report(self) -> pva.Report:
        """
        Gets the pva.Report that is used in the Reptil methods. Also
        used to set the value of the report.

        Returns
        -------
        pva.Report
            pva.Report that is used in the Reptil methods
        """
        return self._report


def open_report(
    report: str, engine_run_range: Tuple[int, int] = None
) -> Reptil:
    """
    Opens a pva report given the filepath of a `.pop` report and
    initializes a `Reptil` object with it.

    Parameters
    ----------
    report : str
        Filepath to the .pop report file.

    Returns
    -------
    Reptil
        The `Reptil` object.
    """
    return Reptil(
        _utils.open_report(report), engine_run_range=engine_run_range
    )
