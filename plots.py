from typing import Any, Sequence

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def generate_log_strip_plot(
    data: Sequence[Sequence[float]], xticks: Sequence[Any],
    title: str, xlabel: str, ylabel: str
) -> None:
    """
    Generates a seaborn strip plot based on the given data.
    The y axis is in a logarithmic scale.
    """
    data = [np.log10(column) for column in data]
    sns.stripplot(data, size=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(xticks)), xticks)
    locations = plt.yticks()[0]
    plt.yticks(locations, [round(10 ** location, 5) for location in locations])
    plt.show()


def generate_line_plot(
    args: Sequence[float], values: Sequence[float], title: str,
    xlabel: str, ylabel: str,
) -> None:
    """
    Generates a 2-dimensional matplotlib graph for the given data.
    args and values lists should have the same length.
    """
    plt.figure()
    plt.plot(args, values, '-', markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def show_confusion_matrix(confusion_matrix: Sequence[Sequence[float]]):
    """
    Shows a seaborn heatmap based on the given confusion matrix.
    """
    sns.heatmap(confusion_matrix, annot=True, fmt='')
    plt.xlabel('Przewidziana wartość')
    plt.ylabel('Prawdziwa wartość')
    plt.show()
