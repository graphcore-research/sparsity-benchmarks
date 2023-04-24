# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import pva


def open_report(report_path: str) -> pva.Report:
    """Open a popvision report."""
    return pva.openReport(report_path) 


def plot(
    values: np.ndarray,
    x_axis: np.ndarray = None,
    x_label: str = "Tile",
    y_label: str = "Bytes",
    title: str = "Memory plot"):
    """
    Create a plot.

    Parameters
    ----------
    values: np.ndarray
        Array to plot

    title: str = "Memory plot"
        Plot title (and file name when saved)

    Returns
    -------
    None.
    """

    if x_axis is None:
        x_axis = np.arange(len(values))

    fig, _ = plt.subplots(figsize=(15, 8))
    plt.plot(x_axis, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.savefig(title.replace(" ", "_") + ".png")

    return
