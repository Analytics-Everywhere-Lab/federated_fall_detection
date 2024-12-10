import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fl_fall.dataset import clean_datasets, load_data, read_labels
from fl_fall.task import load_model
from sklearn.model_selection import train_test_split
import json

BUFFER_SIZE = 60000
BATCH_SIZE = 128
window_size = 40
window_step = 10
user_split = True
epochs = 50
result_path = "federated_fall_detection/result/centeralized"


def plot_data_distribution(train_labels, train_users, dataset_name):
    label_df = read_labels()
    if dataset_name == "UpFall":
        label_df_dataset_name = "UP FALL"
    elif dataset_name == "SiSFall":
        label_df_dataset_name = "SIS FALL"

    partitions, counts = np.unique(train_users, return_counts=True)
    x_axis_values = partitions.astype('int').astype("str")
    activities = np.unique(train_labels)
    green_shades = [
        "#ADFF2F",  # Green Yellow
        "#7FFF00",  # Chartreuse
        "#00FF00",  # Lime
        "#00FF7F",  # Spring Green
        "#00FA9A",  # Medium Spring Green
        "#90EE90",  # Light Green
        "#98FB98",  # Pale Green
        "#66CDAA",  # Medium Aquamarine
        "#20B2AA",  # Light Sea Green
        "#32CD32",  # Lime Green
        "#3CB371",  # Medium Sea Green
        "#2E8B57",  # Sea Green
        "#8FBC8F",  # Dark Sea Green
        "#9ACD32",  # Yellow Green
        "#6B8E23",  # Olive Drab
        "#8A9A5B",  # Moss Green
        "#4F7942",  # Fern Green
        "#228B22",  # Forest Green
        "#556B2F",  # Dark Olive Green
        "#006400"  # Dark Green
    ]
    fig, ax = plt.subplots(figsize=(20, 5))

    until_now = np.zeros(partitions.shape[0])
    for activity_id, color in zip(activities, green_shades[:activities.shape[0]]):
        label_name = label_df[label_df[label_df_dataset_name] == activity_id]['ACTIVITY'].values[0]
        temp_users, temp_counts = np.unique(train_users[train_labels == activity_id], return_counts=True)
        new_values = np.zeros(partitions.shape[0])
        for temp_user, temp_count in zip(temp_users, temp_counts):
            new_values[partitions == temp_user] += temp_count
        ax.bar(x_axis_values, new_values, bottom=until_now, label=label_name, color=color)
        until_now += new_values
    # ax.bar(partition_ids, class_2_counts, bottom=class_1_counts, label='Class 2', color='salmon')

    # Set the labels and title
    ax.set_xlabel('Training User IDs')
    ax.set_ylabel('Count')
    ax.set_title('Per Partition Labels Distribution')
    ax.legend(title='Labels', bbox_to_anchor=(1.28, 1), loc='upper right', borderaxespad=0.)

    plt.tight_layout()

    # Show the plot
    plt.savefig(f"{result_path}/{dataset_name}_training_data_distribution.png")


def generate_and_save_images(model, epoch, test_input, dataset_name, sample_numbers=10):
    reconstructed = model(test_input, training=False)

    original = test_input.numpy()
    reconstructed = reconstructed.numpy()

    fig, axs = plt.subplots(sample_numbers, original.shape[2], figsize=(15, sample_numbers * 3))

    for sample_idx in range(sample_numbers):
        for feature_idx in range(original.shape[2]):
            axs[sample_idx, feature_idx].plot(original[sample_idx, :, feature_idx], label='Original', alpha=0.6)
            axs[sample_idx, feature_idx].plot(reconstructed[sample_idx, :, feature_idx], label='Reconstructed',
                                              linestyle='dashed')
            if sample_idx == 0:
                axs[sample_idx, feature_idx].set_title(f'Feature {feature_idx + 1}')
            axs[sample_idx, feature_idx].legend()

    plt.tight_layout()
    plt.savefig(f"{result_path}/{dataset_name}_reconstructed_epoch_{epoch}.png")
    plt.close()


@tf.function
def train_step(model, model_optimizer, data_batch):
    with tf.GradientTape() as tape:
        reconstructed = model(data_batch, training=True)
        loss = tf.keras.losses.MeanSquaredError()(data_batch, reconstructed)

    gradients = tape.gradient(loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def evaluate_model(model, test_dataset, epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    batch_size = None
    for batch_x in test_dataset:
        if batch_size is None:
            batch_size = batch_x.shape[0]
        reconstructed = model(batch_x, training=False)

        batch_loss = tf.keras.losses.MeanSquaredError()(batch_x, reconstructed)

        epoch_loss_avg.update_state(batch_loss)

        if batch_x.shape[0] != batch_size:
            break

    print(f"Epoch {epoch}: Average Testing loss = {epoch_loss_avg.result().numpy()}")
    return epoch_loss_avg.result().numpy().item()


def train(model, model_optimizer, dataset, test_dataset, epochs, dataset_name):
    result = {"centralized_evaluate": []}
    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_size = None
        for data_batch in dataset:
            if batch_size is None:
                batch_size = data_batch.shape[0]
            loss = train_step(model, model_optimizer, data_batch)
            epoch_loss_avg.update_state(loss)
            if data_batch.shape[0] != batch_size:
                break

        epoch_loss_avg = epoch_loss_avg.result().numpy().item()
        print(f"Epoch {epoch + 1}: Average Training loss  = {epoch_loss_avg}")

        test_loss = evaluate_model(model, test_dataset, epoch + 1)
        result['centralized_evaluate'].append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss_avg,
            "test_loss": test_loss
        })

        if epoch % 10 == 9:
            generate_and_save_images(model, epoch + 1, next(iter(test_dataset.take(1))), dataset_name)
            file_name = f"{result_path}/{dataset_name}_model_state_loss_{loss:.5f}_round_{epoch + 1}.weights.h5"
            model.save_weights(file_name)

    with open(f"{result_path}/{dataset_name}_results.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp)


def main():
    # clean_datasets(True)
    dataset_name = "UpFall"  # or "SiSFall"
    ADL_label_index = 5
    if user_split:
        data = load_data(dataset_name=dataset_name, frequancy="50ms", two_class=False, window_size=window_size,
                         user_split=user_split, window_step=window_step, normlize=True, reload=False)
        train_dataset, _, train_labels, _, train_users, _ = data
        train_dataset = train_dataset[train_labels > ADL_label_index]
        train_users = train_users[train_labels > ADL_label_index]
        train_labels = train_labels[train_labels > ADL_label_index]
        plot_data_distribution(train_labels, train_users, dataset_name)
    else:
        data = load_data(dataset_name=dataset_name, frequancy="50ms", two_class=True, window_size=window_size,
                         user_split=user_split, window_step=window_step, normlize=True, reload=False)
        train_dataset, _, train_labels, _ = data

        train_dataset = train_dataset[train_labels == 0]
        train_labels = train_labels[train_labels == 0]

    x_train, x_val = train_test_split(train_dataset, test_size=0.15, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_val).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    model = load_model()
    model_optimizer = tf.keras.optimizers.Adam(1e-4)
    train(model, model_optimizer, train_dataset, test_dataset, epochs, dataset_name)


if __name__ == '__main__':
    main()
