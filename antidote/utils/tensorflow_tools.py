"""
Tools for working with TensorFlow, including

- BalancedAccuracy class
- FDR class
- GPU configuration function
"""

import keras
import logging
from pathlib import Path
import tensorflow as tf

logger = logging.getLogger(__name__)


# fmt: off
@keras.saving.register_keras_serializable()
class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_pred is rounded to the nearest integer (0 or 1)
        y_pred_rounded = tf.round(y_pred)
        y_true_cast = tf.cast(y_true, y_pred_rounded.dtype)

        # Calculate true positives, true negatives, false positives, and false negatives
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cast, 1), tf.equal(y_pred_rounded, 1)), self.dtype))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cast, 0), tf.equal(y_pred_rounded, 0)), self.dtype))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cast, 0), tf.equal(y_pred_rounded, 1)), self.dtype))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_cast, 1), tf.equal(y_pred_rounded, 0)), self.dtype))

        # Update the state of true positives, true negatives, false positives, and false negatives
        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self) -> float:
        recall_pos = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        recall_neg = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        balanced_acc = (recall_pos + recall_neg) / 2
        return balanced_acc

    def reset_state(self) -> None:
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


@keras.saving.register_keras_serializable()
class FalseDiscoveryRate(tf.keras.metrics.Metric):
    def __init__(self, name='fdr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_pred = tf.round(y_pred)
        true_pos = tf.equal(tf.cast(y_true, tf.bool), tf.cast(y_pred, tf.bool)) & tf.cast(y_pred, tf.bool)
        false_pos = tf.not_equal(tf.cast(y_true, tf.bool), tf.cast(y_pred, tf.bool)) & tf.cast(y_pred, tf.bool)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_pos, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_pos, self.dtype)))

    def result(self) -> float:
        fdr = self.false_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        return fdr

    def reset_state(self) -> None:
        self.true_positives.assign(0)
        self.false_positives.assign(0)


# fmt: on
def configure_gpus(cpu_only=False):
    """
    Configure tensorflow's set_memory_growth behavior.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    cpus = tf.config.experimental.list_physical_devices("CPU")

    if cpu_only:
        tf.config.experimental.set_visible_devices(devices=cpus, device_type="CPU")
        return

    if gpus:
        try:
            # See SO #34199233
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(e)


def load_model(path):
    """
    Handles loading of either a Keras or TF model.
    """
    model = None
    path = Path(path)

    if path.is_file() and path.suffix == ".keras":
        pass
    elif path.is_file():
        path = path.parent

    try:
        model = keras.saving.load_model(
            path,
            custom_objects={
                "BalancedAccuracy": BalancedAccuracy,
                "FalseDiscoveryRate": FalseDiscoveryRate,
            },
        )
    except OSError:
        logger.error("Model format unrecognized. Antidote expects a directory containing a SavedModel file (saved_model.pb)")

    return model
