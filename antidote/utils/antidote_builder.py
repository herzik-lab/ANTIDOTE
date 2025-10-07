from datetime import datetime
import json
from keras.callbacks import TensorBoard
import keras
import keras_tuner as kt
import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path
import random
import requests
import tensorflow as tf
from antidote.utils import tensorflow_tools

logger = logging.getLogger(__name__)


class AntidoteDataGenerator(keras.utils.Sequence):
    """
    Generator based on inheritance from a Keras Sequence object.
    """

    def __init__(self, input_path, size=32, indices=None, key="data", shuffle=False):
        self.input_path = input_path
        self.batch_size = size
        self.key = key
        self.draws = 0

        with pd.HDFStore(input_path, "r") as store:
            self.nrows = store.get_storer(key).nrows

        self.indices = indices if indices is not None else np.arange(self.nrows)
        self.indices_original = self.indices.copy()
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        logger.debug("on_epoch_end called")
        if self.shuffle:
            np.random.shuffle(self.indices_original)
        self.indices = self.indices_original.copy()
        self.draws = 0

    def __getitem__(self, index):
        """
        Keras expects the getitem keyword to be implemented. It also passes an index
        variable to this method, which isn't necessary due to our own implementation
        of index handling.
        """
        logger.debug(f"__getitem__ called: index {index}")
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        logger.debug("__len__ called!")
        return int(math.ceil(self.nrows / self.batch_size))

    def __next__(self):
        """
        Using this as a true generator can be useful in addition to the getitem accession.
        """
        self.draws += 1
        logger.debug(f"__next__ called: {self.input_path}, draw {self.draws}")
        total_samples = len(self.indices_original)
        start_sample = (self.draws - 1) * self.batch_size
        end_sample = min(self.draws * self.batch_size, total_samples)

        # Check if we need to start a new epoch
        if self.draws > 1 and len(self.indices) == 0:
            self.on_epoch_end()
            return self._load_batch()

        # Load and log the current batch
        batch = self._load_batch()
        logger.info(f"Samples {start_sample} - {end_sample} of {total_samples} " f"drawn as chunk {self.draws}.")

        # Start new epoch if we've used all samples
        if end_sample >= total_samples:
            self.on_epoch_end()

        return batch

    def _load_batch(self) -> pd.DataFrame:
        if len(self.indices) < self.batch_size:
            batch_size = len(self.indices)
        else:
            batch_size = self.batch_size

        # Randomly select indices for the chunk and remove from indices
        self.indices, batch_indices = split_indices(self.indices, batch_size)

        logger.info(f"Loaded random chunk of size {batch_size}")

        # Select the chunk based on random indices
        batch_indices.sort()  # speed up accession
        batch = pd.read_hdf(self.input_path, key=self.key, where=pd.Index(batch_indices))

        X = batch.select_dtypes(include=[np.number])
        y = batch["Label"]

        return X, y


class AntidoteHyperModel(kt.HyperModel):
    """
    This model defines the search space for an Antidote hyperparameter search.
    Arguments:
    - input_shape (int): The width of the input data, used to determine the size of the input and hidden layers.
    - mha (bool): Whether to use Multi-Head Attention layer. Defaults to True.
    """

    def __init__(self, input_shape, mha=False) -> None:
        self.input_shape = input_shape
        self.path = None
        self.project_name = None
        self.use_mha = mha

        if mha:
            logger.info("Using Multi-Head Attention...")

    def build(self, hp):
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.4, step=0.1)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        batch_norm = hp.Boolean("batch_norm")
        l2_strength = hp.Float("l2_strength", min_value=1e-5, max_value=1e-2, sampling="log")
        regularizer = keras.regularizers.l2(l2_strength)

        # Add MHA if desired
        if self.use_mha:
            key_dim = hp.Int("key_dim", min_value=16, max_value=64, step=16)
            num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
            embed_dim = key_dim * num_heads

        inputs = keras.layers.Input(shape=(self.input_shape,))

        if self.use_mha:
            input_r = keras.layers.Reshape((self.input_shape, 1))(inputs)
            input_r = keras.layers.Dense(embed_dim)(input_r)
            attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(
                input_r, input_r
            )
            output = keras.layers.LayerNormalization(epsilon=1e-6)(attention)
            output = keras.layers.Flatten()(output)
        else:
            # placeholder to skip MHA
            output = inputs

        # MLP
        nn = keras.layers.Dense(self.input_shape, kernel_regularizer=regularizer)(output)
        if batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation("relu")(nn)
        nn = keras.layers.Dropout(dropout_rate)(nn)

        nn = keras.layers.Dense(self.input_shape, kernel_regularizer=regularizer)(nn)
        if batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation("relu")(nn)
        nn = keras.layers.Dropout(dropout_rate)(nn)

        nn = keras.layers.Dense(self.input_shape // 2, kernel_regularizer=regularizer)(nn)
        if batch_norm:
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation("relu")(nn)

        outputs = keras.layers.Dense(1, activation="sigmoid")(nn)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryCrossentropy(name="cross entropy"),
                keras.metrics.MeanSquaredError(name="Brier score"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.FalsePositives(name="fp"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
                keras.metrics.AUC(name="prc", curve="PR"),
                tensorflow_tools.BalancedAccuracy(),
                tensorflow_tools.FalseDiscoveryRate(),
            ],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [256, 512]),
            **kwargs,
        )


def perform_hyperparameter_search(
    input_path,
    chunksize=None,
    factor=3,
    hdf5_key="data",
    hyperband_iterations=1,
    max_epochs=50,
    num_models=None,
    objective="auc",
    output_path="antidote_tuner",
    use_mha=False,
    validation_fraction=0.2,
    words=True,
):
    """
    Runs a hyperparameter search on an Antidote dataset.
    """
    with pd.HDFStore(input_path, "r") as store:
        row = store.select(hdf5_key, start=0, stop=1)
        X_width = row.shape[1] - 1  # ignore labels column
        X_length = store.get_storer(hdf5_key).nrows
        if not chunksize:
            chunksize = X_length  # use all the data if chunksize isn't specified

    logger.info(f"Running Keras Tuner with chunksize of {chunksize}...")

    # Configure Keras HyperModel
    hm = AntidoteHyperModel(X_width, mha=use_mha)
    hm.log_dir = f"{Path(output_path).name}_tuner_log"
    hm.project_name = f"{Path(output_path).name}_tuner"

    # Set objective based on parameter
    if objective == "auc":
        kt_objective = kt.Objective("val_auc", direction="max")
    elif objective == "acc":
        kt_objective = kt.Objective("val_accuracy", direction="max")
    else:
        logger.error(f"Objective {objective} unrecognized, use `auc` or `acc`. Defaulting to auc...")
        kt_objective = kt.Objective("val_auc", direction="max")

    tuner = kt.Hyperband(
        hm,
        objective=kt_objective,
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        directory="./",
        project_name=hm.project_name,
        # distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    tensorboard_callback = TensorBoard(log_dir=hm.log_dir, histogram_freq=20)

    if objective == "auc":
        early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", patience=5)
    elif objective == "acc":
        early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)
    else:
        logger.error(f"Objective {objective} unrecognized, use `auc` or `acc`. Defaulting to auc...")
        early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", patience=5)

    # In this case, we'll use the AntidoteDataGenerator to return a single chunk instead of passing the generator
    # directly to keras tuner.
    train_indices, val_indices = split_indices(np.arange(X_length), math.floor(X_length * validation_fraction))

    gr_train = AntidoteDataGenerator(
        input_path,
        size=(math.floor(chunksize * (1 - validation_fraction))),
        indices=train_indices,
    )
    gr_val = AntidoteDataGenerator(
        input_path,
        size=(math.floor(chunksize * validation_fraction)),
        indices=val_indices,
    )

    train_chunk = next(gr_train)
    val_chunk = next(gr_val)

    tuner.search(
        train_chunk[0],
        train_chunk[1],
        epochs=20,
        validation_data=(val_chunk[0], val_chunk[1]),
        callbacks=[tensorboard_callback, early_stop],
    )

    # Save all or specified number of trials
    if num_models is None:
        num_models = len(tuner.oracle.trials)
    trials = tuner.oracle.get_best_trials(num_trials=num_models)
    models = tuner.get_best_models(num_models=num_models)

    # Save all the models with a random name if possible
    if words:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(
                "https://rayberkeley.com/posts/words.txt",
                headers=headers,
            )
            if response.status_code == 200:
                word_list = response.text.splitlines()
            else:
                word_list = None
        except Exception:
            word_list = None

    for index, model in enumerate(models):
        base_path = Path(f"{output_path}_tuner_best_models/model_{index}")
        model_path = base_path
        if word_list:
            word = random.choice(word_list)
            model_path = base_path.with_name(f"{base_path.stem}_{word}{base_path.suffix}")
        model.save(model_path)

    return models[0], trials[0] if trials else None


def split_indices(indices, size) -> tuple[np.array, np.array]:
    """
    A general function to take an input array and return two subarrays with a random split of the input data.
    This is helpful for splitting indices for training and testing in a general input dataset.

    Args:
    - indices (np.array): The input array to be split
    - num (int):          The number of samples to draw from the array
    """
    # validation data, specified using a validation fraction parameter
    minor_indices = np.random.choice(indices, size=size, replace=False)
    # the leftover data is the training data
    major_indices = np.setdiff1d(indices, minor_indices)

    return major_indices, minor_indices


def append_metrics_to_file(metrics_file_path, epoch, chunk, phase, metrics):
    """A quick function to record metrics related to training progress."""
    metric_names = [
        "loss",
        "cross entropy",
        "Brier score",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "auc",
        "prc",
        "balanced_accuracy",
        "false_discovery_rate",
    ]

    # Check if the length of metrics matches the number of names
    if len(metrics) != len(metric_names):
        raise ValueError("The number of provided metrics does not match the number of metric names.")

    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Integrate the metrics directly into the entry dictionary
    entry = {
        "timestamp": current_time,
        "epoch": epoch,
        "chunk": chunk,
        "phase": phase,
    }
    entry.update({metric_names[i]: metrics[i] for i in range(len(metric_names))})

    # Append the entry to the specified file
    with open(metrics_file_path, "a+") as f:
        json.dump(entry, f)
        f.write("\n")


def train_model(
    model,
    input_path,
    epochs=20,
    output_path="antidote_train",
    hdf5_key="data",
    batch_size=256,  # populated by Keras tuner or the user
    chunksize=None,
    validation_fraction=0.2,
    testing_path=None,
):
    """
    Continues training a model generated by Keras tuner.
    """
    with pd.HDFStore(input_path, "r") as store:
        X_length = store.get_storer(hdf5_key).nrows
        if not chunksize:
            chunksize = X_length

    # split the indices for the full dataset
    train_indices, val_indices = split_indices(np.arange(X_length), math.floor(X_length * validation_fraction))

    gr_train = AntidoteDataGenerator(
        input_path,
        size=(math.floor(chunksize * (1 - validation_fraction))),
        indices=train_indices,
        shuffle=False,
    )
    gr_val = AntidoteDataGenerator(
        input_path,
        size=(math.floor(chunksize * validation_fraction)),
        indices=val_indices,
        shuffle=False,
    )

    test_generators = {}
    if testing_path:
        for path in testing_path:
            test_generator = AntidoteDataGenerator(
                path,
                size=chunksize,
                shuffle=False,
            )

            pathname = Path(path).name
            test_generators[pathname] = test_generator

    chunks_processed = 0
    start_from_chunk = 100
    patience = 10
    min_delta = 0.001

    best_val_loss = float("inf")
    patience_counter = 0

    metrics_file_path = Path(f"{output_path}_checkpoints/metrics.json")
    metrics_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing metrics to {metrics_file_path}")

    for epoch in range(epochs):
        logger.info(f"Running epoch {epoch} of {epochs}...")
        for train_chunk, val_chunk in zip(gr_train, gr_val):
            logger.debug("Started fit")

            if (train_chunk[0] is None and train_chunk[1] is None) or (val_chunk[0] is None and val_chunk[1] is None):
                break

            history = model.fit(
                train_chunk[0],
                train_chunk[1],
                batch_size=batch_size,
                epochs=1,
                validation_data=(val_chunk[0], val_chunk[1]),
                verbose=1,
            )

            current_val_loss = history.history["val_loss"][0]

            train_metrics = model.evaluate(train_chunk[0], train_chunk[1], batch_size=batch_size, verbose=0)
            append_metrics_to_file(metrics_file_path, epoch, chunks_processed, "train", train_metrics)
            val_metrics = model.evaluate(val_chunk[0], val_chunk[1], batch_size=batch_size, verbose=0)
            append_metrics_to_file(metrics_file_path, epoch, chunks_processed, "val", val_metrics)

            logger.debug(f"Chunk {chunks_processed} - val_loss: {current_val_loss}")

            if testing_path:
                for pathname, test_generator in test_generators.items():
                    test_chunk = next(test_generator)
                    test_metrics = model.evaluate(test_chunk[0], test_chunk[1], batch_size=batch_size, verbose=0)
                    append_metrics_to_file(
                        metrics_file_path,
                        epoch,
                        chunks_processed,
                        f"test_{pathname}",
                        test_metrics,
                    )

            # early stopping
            if chunks_processed >= start_from_chunk:
                if best_val_loss - current_val_loss > min_delta:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    logger.debug("Improved validation loss; resetting patience counter.")
                else:
                    patience_counter += 1
                    logger.debug(f"No sufficient improvement. Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at chunk {chunks_processed}.")
                    break

            chunks_processed += 1
            logger.debug(f"Chunk {chunks_processed} processed.")

            tf.keras.backend.clear_session()

        # break out of the outer loop if early stopping was triggered
        if chunks_processed >= start_from_chunk and patience_counter >= patience:
            break

    model.save(Path(f"{output_path}_final_model"))
    logger.info(f"Training complete. Model saved as {output_path}_final_model.")

    return model
