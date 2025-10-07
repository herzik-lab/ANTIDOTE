"""
Returns a RelionClass3DJob Object with a balanced T/F data attribute.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def balance(job, num: int, force_balance=True, replace=False):
    """
    Balances the data attribute in a class3D job.

    Args:
    -   num (int): The number of true and false particles to return.
    -   force_balance (bool): Prevents imbalances if the requested balance is larger than the number
                              of available particles. For example, if a job has 90 T and 10 F particles,
                              and balance() is called with 50 particles, force_balance will return 10 T
                              and 10 F particles while not force_balance will return 50 T and 10 F.
    -   replace (bool): Samples the data with replacement. This is useful for highly imbalanced data
                        with small numbers of particles, but is generally not advised. An implementation
                        that only replaces when all particles have been exhausted would be more useful.
    """

    data = job.data.copy()

    label_counts = data[("Label", "")].value_counts()

    # it may be useful to specify ratio later
    num_true, num_false = num, num
    count_true, count_false = label_counts.get(True, 0), label_counts.get(False, 0)

    if force_balance:
        min_count = min(num_true, num_false, count_true, count_false)
        if min_count < num:
            logger.warning(
                f"{job.name} balancing adjusted to {min_count} samples per label instead of {num} due to insufficient samples."
            )
        num_true, num_false = min_count, min_count
    else:
        num_true = min(num_true, count_true)
        num_false = min(num_false, count_false)
        if num_true < num or num_false < num:
            logger.warning(
                f"{job.name} balancing adjusted to {num_true} True samples and {num_false} False samples instead of {num} due to insufficient samples."
            )

    true_rows = data[data[("Label", "")] == True].sample(num_true, replace=replace)
    false_rows = data[data[("Label", "")] == False].sample(num_false, replace=replace)

    balanced_data = pd.concat([true_rows, false_rows])

    job.data = balanced_data

    return job
