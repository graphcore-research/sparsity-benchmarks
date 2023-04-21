import pva
from collections import defaultdict
from ._reptil_namespace import ReptilNamespace
import fnmatch as fnm


class Cycles(ReptilNamespace):
    """
    Namespace for cycles metrics utility functions
    """

    def intervals(self, vertices: list = [], merge_stream_copy_segments: bool = True) -> dict:
        """
        Extract the measured exchange, stream copy and compute cycle intervals
        for a list of vertices.

        Parameters
        ----------
        vertices : list, optional
            A list of strings where each string corresponds to a vertex name.
            vertex names can use unix shell-style wildcards.
        merge_stream_copy_segments: bool, optional
            Whether to return StreamCopy segments (Begin, Mid and End) merged into
            one interval e.g. [StreamCopyBegin.start, StreamCopyEnd.end] or as
            independent intervals.

        Returns
        -------
        dict
            A dict with three elements, each a list of cycle intervals for
            compute, stream copy and exchange phases.
        """

        # Execution steps for the first run, this would ideally be the second
        # run as it would then not have the chance of being affected by initial
        # warmup time on the IPU. However, there is an issue that there is a
        # large stream copy begin as the host deals with the instrumentation
        # data that will massively scew the reported cycles. This would
        # hopefully be fixed by T44614
        steps = self._steps()
        computeIntervals = []
        streamCopyIntervals = []
        doExchangeIntervals = []

        class IntervalVisitor(pva.ProgramVisitor):
            streamCopyStart = 0
            streamCopyMid = False

            def __init__(self, cyclesFrom, cyclesTo, vertices, merge_stream_copy_segments):
                self.cyclesFrom = cyclesFrom
                self.cyclesTo = cyclesTo
                self.vertices = vertices
                self.merge_stream_copy_segments = merge_stream_copy_segments
                super(IntervalVisitor, self).__init__()

            def visitOnTileExecute(self, onTileExecute):
                if (not self.vertices or any(fnm.fnmatch(onTileExecute.name, prog) for prog in self.vertices)):
                    computeIntervals.append(
                        [self.cyclesFrom.max, self.cyclesTo.max, onTileExecute.name])

            def visitDoExchange(self, doExchange):
                doExchangeIntervals.append(
                    [self.cyclesFrom.max, self.cyclesTo.max])

            def visitStreamCopyBegin(self, begin):
                if not self.merge_stream_copy_segments:
                    streamCopyIntervals.append([self.cyclesFrom.max, self.cyclesTo.max, "StreamCopyBegin"])
                else:
                    IntervalVisitor.streamCopyStart = self.cyclesFrom.max

            def visitStreamCopyMid(self, mid):
                if not self.merge_stream_copy_segments:
                    streamCopyIntervals.append([self.cyclesFrom.max, self.cyclesTo.max, "StreamCopyMid"])
                else:
                    IntervalVisitor.streamCopyMid = (
                        self.cyclesTo.max - self.cyclesFrom.max) != 0

            def visitStreamCopyEnd(self, end):
                if not self.merge_stream_copy_segments:
                    streamCopyIntervals.append([self.cyclesFrom.max, self.cyclesTo.max, "StreamCopyEnd"])
                else:
                    if (IntervalVisitor.streamCopyMid):
                        streamCopyIntervals.append(
                            [IntervalVisitor.streamCopyStart, self.cyclesTo.max])
                    IntervalVisitor.streamCopyMid = False

        for step in steps:
            ipu = step.ipus[0]
            cycle_from = ipu.activeCycles.cyclesFrom
            cycle_to = ipu.activeCycles.cyclesTo
            visitor = IntervalVisitor(cycle_from, cycle_to, vertices, merge_stream_copy_segments)
            step.program.accept(visitor)

        return {"compute": computeIntervals,
                "stream_copies": streamCopyIntervals,
                "exchange": doExchangeIntervals}


    def program(self) -> dict:
        """
        Extract the measured cycles aggregated by program

        returns
        -------
        dict
            program type (str) -> total cycles (int)
        """

        report = self._report
        cyclesByType = defaultdict(int)

        for s in self._steps():
            cyclesByType[s.program.type.name] += sum(s.cyclesByTile)

        return cyclesByType

    def computeset_tile(self) -> dict:
        """
        Extract the measured cycles aggregated by compute set and split by tile

        returns
        -------
        dict
            compute set name (str) -> list of cycles, one entry per tile [(int)]
        """


        cyclesBySetTile = defaultdict(list)

        for s in self._steps():
            for c in s.computeSets:
                if not len(cyclesBySetTile[c.name]):
                    cyclesBySetTile[c.name] = c.cyclesByTile
                else:
                    cyclesBySetTile[c.name] = [sum(x) for x in zip(
                        cyclesBySetTile[c.name], c.cyclesByTile)]

        return cyclesBySetTile

    def computeset(self) -> dict:
        """
        Extract the measured cycles aggregated by compute set

        returns
        -------
        dict
            compute set (str) -> total cycles (int)
        """

        cyclesBySet = defaultdict(int)

        for s in self._steps():
            for c in s.computeSets:
                cyclesBySet[c.name] += sum(c.cyclesByTile)

        return cyclesBySet

    def vertex(self) -> dict:
        """
        Extract the measured cycles aggregated by vertex
        (not yet supported in Poplar)

        returns
        -------
        dict
            vertex name (str) -> total cycles (int)
        """

        raise NotImplementedError("Not yet supported")

        cyclesByVertex = defaultdict(int)

        for s in self._steps():
            for c in s.computeSets:
                for v in c.vertices:
                    cyclesByVertex[v.type.name] += v.estimatedCycles

        return cyclesByVertex
