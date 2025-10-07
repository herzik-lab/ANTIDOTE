"""
This function returns a recommended cutoff threshold based on a KDE of the ANTIDOTE assignments. For now,
this just looks at the global minimum, which assumes a bimodal distribution.

Args:
-   sf (starfile): The starfile object, with particle assignments from antidote.
-   label_field_name (str): The name of the column in the starfile particles data that contains antidote
                            labels.

Returns:
-   threshold (float): A recommended threshold value.

Limitations:
A better method for determining the threshold that doesn't make so many assumptions about the shape of the
data may be preferred here.
"""

from KDEpy import FFTKDE
import numpy as np
from scipy.signal import find_peaks, peak_widths
import starfile


def calculate_kde(input_array: np.array) -> np.array:
    """
    Calculate a KDE on an input array, and return x, y coordinates for points along the KDE. For now, this
    approach uses the ISJ method described in Botev, Grotowski, & Kroese. Ann. Stat. 2010. A simpler
    approach may suffice later.
    """

    x, y = FFTKDE(kernel="gaussian", bw="ISJ").fit(input_array).evaluate()

    return x, y


def calculate_threshold(x: np.array, y: np.array) -> float:
    """
    Calculate a recommended threshold based on the angle between the peak maxima and the left side of the peak
    at 2/3 peak height. The recommended threshold is the point that intersects the line that runs between these
    two points at the baseline. This generally seems to capture the majority of the volume of the peak that
    represents picks with a high confidence.

    Args:
    -   x (np.array): The x axis of the input KDE.
    -   y (np.array): The y axis of the input KDE.

    Returns:
    -   (float): the point of intersection between the line and the baseline, which serves as an approximate
                 threshold for Antidote labels.
    """
    prominence = np.std(y)
    height = 2 * y.mean()
    peaks, indices = find_peaks(y, prominence=prominence, height=height)

    # get the x and y coordinates of the maximia of the rightmost peak
    x1 = peaks.max()
    y1 = y[int(x1.round())]

    # get the x and y coordinates of the rightmost peak at 1/3 peak height
    x2 = peak_widths(y, peaks, rel_height=0.666)[2].max()
    y2 = y[int(x2.round())]

    # find the slope and intercept at the x-axis
    slope = np.polyfit([x1, x2], [y1, y2], 1)
    intercept = -slope[1] / slope[0]

    recommended_threshold = x[int(intercept.round())].round(2)

    return max(0.05, min(0.95, recommended_threshold))


def calculate_threshold_bimodal(x: np.array, y: np.array) -> float:
    """
    Note: this method has been superseded by the peak volume-based method in calculate_threshold()

    Calculate a recommended threshold based on the minima of the KDE between the two most prominent peaks in
    the KDE. For an effective binary classification, this will avoid the minima on the extreme ends of the
    KDE while setting the recommended threshold in between the two populations of true and false peaks. Note
    that this approach may fail if there are more than two major groups in the KDE.

    Args:
    -   x (np.array): The x axis of the input KDE.
    -   y (np.array): The y axis of the input KDE.

    Limitations:
    This function assumes that the classification is binary, and that the ideal threshold sits at the minimum
    value between each peak.
    """
    prominence = np.std(y)
    height = 2 * y.mean()
    peaks, indices = find_peaks(y, prominence=prominence, height=height)

    # get the index of the maxima in the KDE
    indices_of_bimodal_maxima = peaks[np.argsort(indices["prominences"])[-2:]]

    # slice the x and y KDE values to remove minima at boundaries
    x = x[indices_of_bimodal_maxima.min() : indices_of_bimodal_maxima.max()]
    y = y[indices_of_bimodal_maxima.min() : indices_of_bimodal_maxima.max()]

    # find the x value that represents the smallest value of y
    recommended_threshold = x[y.argmin()].round(2)

    return max(0.05, min(0.95, recommended_threshold))


def run(sf: starfile, label_field_name: str) -> int:
    labels = np.array(sf["particles"][label_field_name])

    x, y = calculate_kde(labels)
    recommended_threshold = calculate_threshold(x, y)

    return recommended_threshold
