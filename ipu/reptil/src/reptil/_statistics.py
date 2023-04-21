import numpy as np
from ._utils import plot


def summary_stats(arr: np.ndarray) -> dict:
    """
    Get summary statistics for a given numerical array.
    
    Parameters
    ----------
    arr: np.ndarray
        Array from which to calculate statistics

    Returns
    -------
    stats: dict
        Dict of named statistics calculated from arr
    """

    stats = {}

    stats['max'] = np.max(arr)
    stats['min'] = np.min(arr)
    stats['mean'] = np.mean(arr)
    stats['median'] = np.percentile(arr, 50)
    stats['std'] = np.std(arr)

    return stats


def tile_balance(
    tile_memory: np.ndarray,
    bytes_per_tile: int = 624*1024
    ) -> float:
    """
    Calculate tile balance across an IPU.

    Notes
    -----
    "balance" here is a measure how well spread the memory is over all tiles. 
    In general, this is some reduction of a similarity measure between 
    individual tile memory usasge values and a representative statistics for 
    all those values. Specifically in this case, its the square  root of the
    mean of the squared differences between the mean tile memory usage and the
    individual values (RMSE).

    This value is then scaled by the maximum possible rmse (which exists 
    because tile memory usage is bound between 0 and 624KB[1]) to give a value
    between 0 and 1 which is much more interpretable.

    This is so:
        - The algorithm is resilient to heavily skewed tile memory 
        distributions that average out well
        - Tile values that are far from the mean are represented stronger
        - the value is easy to interpret

    References
    ----------
    [1]: https://docs.graphcore.ai/projects/ipu-overview/en/latest/about_ipu.
        html#gc200-mk2-ipu-memory

    Parameters
    ----------
    tile_memory: np.ndarray
        Memory values for each tile in an IPU. 

    bytes_per_tile: int = 624*1024
        Bytes available per tile, defaults to MK2 specification

    Returns
    -------
    balance: float
        A measure from 0 to 1 (low to high) of how well balanced the memory
        across all tiles on an IPU are
    """

    mean = np.mean(tile_memory)

    # Worst case changes depending on whether baseline for error is below
    # or above halfway point
    if mean <= bytes_per_tile/2:
        worst_case_usage = np.ones(len(tile_memory))*bytes_per_tile
    else:
        worst_case_usage = np.zeros(len(tile_memory))
    
    # Scale actual error by the maximum possible error and subtract from max
    root_mean_square = lambda x : np.sqrt(np.mean(np.square(x)))
    error = root_mean_square(tile_memory - mean)
    max_error = root_mean_square(worst_case_usage - mean)

    # In case somehow the code has achieved perfect balance
    if max_error == 0:
        balance = 1
    else:
        balance = 1 - error/max_error

    return balance


def histogram(values: np.ndarray, title: str = 'Memory histogram'):
    """
    Plot a histogram from a set of values.

    Parameters
    ----------
    values: np.ndarray
        Array of values to create a histogram from

    title: str = 'Memory histogram'
        Title for the plot

    Returns
    -------
    None.
    """

    histogram, bin_edges = np.histogram(values)
    plot(histogram, bin_edges[1:], "Bytes", "Frequency", title)

    return
