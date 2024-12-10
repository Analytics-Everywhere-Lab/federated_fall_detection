"""fl-fall: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import tensorflow as tf
from fl_fall.task import load_data, load_model, get_optimizer


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
            self, model, optimizer, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.train_dataset, self.val_dataset = data
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def train_step(self, model, model_optimizer, data_batch):
        with tf.GradientTape() as tape:
            reconstructed = model(data_batch, training=True)
            loss = tf.keras.losses.MeanSquaredError()(data_batch, reconstructed)

        gradients = tape.gradient(loss, model.trainable_variables)
        model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def evaluate_model(self, model, test_dataset):
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_size = None
        sample_counts = tf.Variable(0)
        for batch_x in test_dataset:
            if batch_size is None:
                batch_size = batch_x.shape[0]
            reconstructed = model(batch_x, training=False)

            batch_loss = tf.keras.losses.MeanSquaredError()(batch_x, reconstructed)
            epoch_loss_avg.update_state(batch_loss)

            sample_counts.assign_add(batch_x.shape[0])

            if batch_x.shape[0] != batch_size:
                break

        return epoch_loss_avg.result().numpy().item(), sample_counts.numpy().item()

    def train(self, model, model_optimizer, dataset):
        sample_counts = tf.Variable(0)
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_size = None
        for data_batch in dataset:
            if batch_size is None:
                batch_size = data_batch.shape[0]
            loss = self.train_step(model, model_optimizer, data_batch)
            epoch_loss_avg.update_state(loss)
            sample_counts.assign_add(data_batch.shape[0])
            if data_batch.shape[0] != batch_size:
                break
        return sample_counts.numpy().item()

    def fit(self, parameters, config):
        self.model.set_weights(parameters.copy())
        for _ in range(self.epochs):
            sample_counts = self.train(self.model, self.optimizer, self.train_dataset)
        return self.model.get_weights(), sample_counts, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, sample_counts = self.evaluate_model(self.model, self.val_dataset)
        return loss, sample_counts, {}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    optimizer = get_optimizer()
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, optimizer, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
