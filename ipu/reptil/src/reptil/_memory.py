import numpy as np
from ._reptil_namespace import ReptilNamespace
from ._statistics import summary_stats, tile_balance


class Memory(ReptilNamespace):
    """
    Namespace for memory utility functions
    """

    class AggregationLevels:
        """
        Small class for handling return types of memory categories
        """

        def __init__(self, memory_array: np.ndarray, num_ipus: int):
            self.memory_array = memory_array
            self.num_ipus = num_ipus
            return

        @property
        def ipus(self) -> np.ndarray:
            """
            Memory usage per IPU.
            """
            # Reshape array of all tiles by IPU, and sum across each IPU
            memory = np.sum(
                np.reshape(self.memory_array, (self.num_ipus, -1)),
                axis=1
            )
            return memory

        @property
        def tiles(self) -> np.ndarray:
            """
            Memory usage per tile for each IPU.
            """
            # Reshape array of all tiles by IPU
            memory = np.reshape(self.memory_array, (self.num_ipus, -1))
            return memory

        @property
        def total(self) -> int:
            """
            Total memory usage from profiled application.
            """
            memory = int(np.sum(self.memory_array))
            return memory

    @property
    def vertex(self) -> AggregationLevels:
        """
        Get total vertices memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Vertex memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        # Iterate through each vertex in each tile (expensive)
        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.category.vertexCode.total

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def exchange(self) -> AggregationLevels:
        """
        Get total exchange code memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Exchange code memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            category = tile.memory.category
            memory_array[i] = sum([
                category.internalExchangeCode.total,
                category.globalExchangeCode.total,
                category.hostExchangeCode.total])

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def constants(self) -> AggregationLevels:
        """
        Get total constants memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Constants memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        # Iterate through each vertex in each tile (expensive)
        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.category.constant.total

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def control(self) -> AggregationLevels:
        """
        Get total control code memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Control code memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.category.controlCode.total

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def always_live(self) -> AggregationLevels:
        """
        Get total always live memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Always live memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.alwaysLiveBytes

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def not_always_live(self) -> AggregationLevels:
        """
        Get total not always live memory (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Not always live memory usage in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.notAlwaysLiveBytes

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def including_gaps(self) -> AggregationLevels:
        """
        Get total memory including gaps (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Total memory usage including gaps in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.total.includingGaps

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def excluding_gaps(self) -> AggregationLevels:
        """
        Get total memory excluding gaps (in Bytes) on each tile for each IPU.

        Returns
        -------
        memory: AggregationLevels
            Total memory usage excluding gaps in a AggregationLevels object
        """

        memory_array = np.zeros(len(self._report.compilation.tiles))

        for i, tile in enumerate(self._report.compilation.tiles):
            memory_array[i] = tile.memory.total.excludingGaps

        # Return the memory in the requested format
        num_ipus = len(self._report.compilation.ipus)
        memory = self.AggregationLevels(memory_array, num_ipus)

        return memory

    @property
    def total(self) -> AggregationLevels:
        """Alias for including_gaps."""
        return self.including_gaps

    @property
    def peak_liveness(self) -> int:
        """
        Get the peak total liveness value of the run over all steps.

        Returns
        -------
        peak_liveness: int
            The sum of the always live memory and maximum not always live
            memory
        """

        # Iterate over all program steps and keep maximum not always live
        max_value = 0
        for step in self._report.compilation.livenessProgramSteps:
            value = step.notAlwaysLiveMemory.bytes
            max_value = max(max_value, value)

        # livenessProgramSteps gives information per replica so this must be multiplied by num_of_replicas to get total
        not_always_live_memory_peak = max_value * self._report.compilation.target.numReplicas

        peak_liveness = self.always_live.total + not_always_live_memory_peak

        return peak_liveness

    @property
    def peak_liveness_proportion(self) -> float:
        """
        Get peak_liveness as a proportion of total available memory.

        Notes
        -----
        This metric is just dynamically scaled peak liveness. As the peak
        liveness value is only useful to know if you know the total IPU memory
        available to a program, here the peak liveness is divided by the number
        of ipus multiplied by the total memory available per ipu. This gives a
        number between 0 and 1, and in cases where the model is OOM, greater
        than 1.

        Returns
        -------
        peak_liveness_proportion: float
            The proportion of all memory available to the program that was used
            by the program at any given point
        """

        peak_liveness = self.peak_liveness
        num_ipus = len(self._report.compilation.ipus)
        bytes_per_ipu = self._report.compilation.target.bytesPerIPU

        peak_liveness_proportion = peak_liveness / (num_ipus * bytes_per_ipu)

        return peak_liveness_proportion

    @property
    def summary(self, tile_level=True) -> dict:
        """
        Calculate summary statistics from a popvision report.

        Parameters
        ----------
        tile_level : bool, optional
            Whether to include tile level statistics in the output dict.

        Returns
        -------
        summary_dict: dict
            Contains high (IPU) and low (tile) level breakdowns of memory usage
            from the popvision report.
        """

        # Get summaries from all IPUs
        summary_dict = {}
        memory = self.total
        num_ipus = len(self._report.compilation.ipus)

        # Get most important metrics
        summary_dict["Peak liveness"] = {
            "total": self.peak_liveness,
            "proportion": self.peak_liveness_proportion
        }

        # Get totals by all categories
        summary_dict["Memory categories"] = {
            "vertex": self.vertex.total,
            "control": self.control.total,
            "exchange": self.exchange.total,
            "constants": self.constants.total,
            "always_live": self.always_live.total,
            "including_gaps": self.including_gaps.total,
            "excluding_gaps": self.excluding_gaps.total,
            "not_always_live": self.not_always_live.total,
        }

        # Get higher level IPU stats
        summary_dict["IPU level"] = summary_stats(memory.ipus)

        # Get more detailed lower level stats for each IPU
        if tile_level:
            tile_dict = {}
            for i in range(num_ipus):
                tile_dict[f"IPU {i}"] = summary_stats(memory.tiles[i])
                tile_dict[f"IPU {i}"]["balance"] = tile_balance(
                    memory.tiles[i],
                    self._report.compilation.target.bytesPerTile
                )

            summary_dict["Tile level"] = tile_dict

        return summary_dict
