import os
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from keras import layers
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from flwr.common.typing import UserConfig

# Assuming the necessary data functions are available
from fl_fall.dataset import load_data as fall_load_data

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

train_dataset, train_labels, unique_train_users, train_users, test_dataset = None, None, None, None, None

BUFFER_SIZE = 60000
BATCH_SIZE = 128
LATENT_SIZE = 32
LAYER_SIZE = 128
TIME_STEPS = 40
N_FEATURES = 6


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def load_model():
    """Load and return the LSTM-based model for federated training"""
    input_layer = layers.Input(shape=(TIME_STEPS, N_FEATURES))
    lstm_1 = layers.LSTM(LAYER_SIZE, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(
        input_layer)
    dropout_1 = layers.Dropout(0.3)(lstm_1)
    latent = layers.LSTM(LATENT_SIZE, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                         name="latent_layer")(dropout_1)
    dropout_2 = layers.Dropout(0.3)(latent)
    lstm_2 = layers.LSTM(LAYER_SIZE, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(
        dropout_2)
    dropout_3 = layers.Dropout(0.3)(lstm_2)
    output_layer = layers.TimeDistributed(layers.Dense(N_FEATURES, activation='linear'))(dropout_3)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    return model


def get_optimizer():
    return tf.keras.optimizers.Adam(1e-4)


def load_data(partition_id, num_partitions):
    """Load and partition the dataset for federated learning."""
    global train_dataset, train_labels, unique_train_users, train_users, test_dataset
    if train_dataset is None:
        # Load the dataset using the provided function
        window_size = 40
        window_step = 10
        data = fall_load_data(dataset_name="SiSFall", frequancy="50ms", two_class=True, window_size=window_size,
                              user_split=True, window_step=window_step, normlize=True, reload=True)
        train_dataset, test_dataset, train_labels, test_labels, train_users, _ = data
        # Filter only the non-fall activities (label == 0) for training
        train_dataset = train_dataset[train_labels == 0]
        test_dataset = test_dataset[test_labels == 0]
        train_users = train_users[train_labels == 0]
        unique_train_users = np.unique(train_users)

    if partition_id == -1:
        # Return Test dataset
        final_test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return final_test_dataset
    else:
        # Partition the data into `num_partitions` parts
        # based on the different user ids
        temp_user_id = unique_train_users[partition_id]
        partitioned_data = train_dataset[train_users == temp_user_id]
        x_train, x_val = train_test_split(partitioned_data, test_size=0.15, random_state=42)
        final_train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        final_test_dataset = tf.data.Dataset.from_tensor_slices(x_val).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        return final_train_dataset, final_test_dataset
