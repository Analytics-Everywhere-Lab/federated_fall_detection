"""fl-fall: A Flower / TensorFlow app."""
from typing import Optional, Union

import flwr as fl
import tensorflow as tf
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl_fall.task import load_model, load_data
from fl_fall.strategy import CustomFedAvg


def gen_evaluate_fn(test_dataset):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model()
        model.set_weights(parameters_ndarrays)
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

        return epoch_loss_avg.result().numpy().item(), {}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.001
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())
    x_test = load_data(-1, None)

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(x_test),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
