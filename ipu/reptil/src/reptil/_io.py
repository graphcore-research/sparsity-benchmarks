from typing import TYPE_CHECKING, Union
import pva
from ._reptil_namespace import ReptilNamespace

if TYPE_CHECKING:
    from reptil import Reptil


class IO(ReptilNamespace):
    """
    Namespace for IO metric utility functions
    """

    class Bandwidth(ReptilNamespace):
        """
        Namespace for IO bandwidth utility functions
        """

        def _calculate_bandwidth(self, prog_type: pva.Program.Type) -> float:
            """
            Calculate the bandwidth associated with a given program type.

            Parameters
            ----------
            prog_type : pva.Program.Type
                The program type which you would like to calculate the
                bandwidth for. Any program that is not a StreamCopyMid,
                GlobalExchange, DoExchange will return 0.

            Returns
            -------
            float
                The bandwidth in bytes per second.
            """
            run = self._report.execution.runs[0]
            total_data = 0
            for step in run.steps:
                ipu_step = step.ipus[0]
                if type(step.program.type) is type(prog_type):
                    total_data += ipu_step.dataIn
                    total_data += ipu_step.dataOut

            duration = run.microseconds.end - run.microseconds.start

            # The duration can be 0 if the run took less than 100ms or when
            # running on the IPU model
            if duration == 0: 
                return 0

            return total_data / (duration / 10**6)

        @property
        def host(self) -> float:
            """
            The bandwidth for host based exchange. This is measured as the
            total data sent in both directions for each stream copy in the
            total execution time.

            Returns
            -------
            float
                Host bandwidth in bytes per second
            """
            return self._calculate_bandwidth(pva.Program.Type.StreamCopyMid)

        @property
        def inter_ipu(self) -> float:
            """
            The bandwidth for inter-ipu exchange. This is measured as the total
            data sent in both directions for each global exchange in the total
            execution time.

            Returns
            -------
            float
                Inter-ipu bandwidth in bytes per second
            """
            return self._calculate_bandwidth(pva.Program.Type.GlobalExchange)

    def __init__(self, parent: Union['Reptil', 'ReptilNamespace']):
        super(IO, self).__init__(parent)
        self._bandwidth = self.Bandwidth(self)

    @property
    def bandwidth(self) -> Bandwidth:
        """
        Namespace for methods related to IO bandwidth.

        Returns
        -------
        Bandwidth
            Object that owns the IO bandwidth methods.
        """
        return self._bandwidth

    def overlapped(self, vertices: list = []) -> dict:
        """
        Return the cycle intervals where the list of vertices provided
        overlapped, or did not overlap with a stream copy

        Parameters
        ----------
        vertices : list, optional
            List of vertices that should be checked to see if they overlapped
            with a stream copy, by default []

        Returns
        -------
        dict
            A dict with two elements; `overlapped` and `non_overlapped` each a
            list of cycle intervals for which `vertices` did or did not overlap
            with some IO
        """
        intervals = self._parent.cycles.intervals(vertices=vertices)
        compute_intervals = intervals["compute"]
        stream_copy_intervals = intervals["stream_copies"]

        overlapped_vertex_intervals = []
        non_overlapped_vertex_intervals = []

        def check_overlap(low1, high1, low2, high2):
            return low1 < high2 and low2 < high1

        def get_overlap(low1, high1, low2, high2):
            overlap = min(high1, high2) - max(low1, low2)
            return overlap

        overlap = 0
        for compute in compute_intervals:
            is_overlapped = False
            for stream in stream_copy_intervals:
                if check_overlap(compute[0], compute[1], stream[0], stream[1]):
                    is_overlapped = True
                    overlap += get_overlap(compute[0],
                                           compute[1], stream[0], stream[1])

            if not is_overlapped and stream_copy_intervals[-1][1] > compute[0]:
                # If the compute set didn't overlap, but isn't the last iteration
                non_overlapped_vertex_intervals.append(compute)
            else:
                overlapped_vertex_intervals.append(compute)

        return {"overlapped": overlapped_vertex_intervals,
                "non_overlapped": non_overlapped_vertex_intervals}
