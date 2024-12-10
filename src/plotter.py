import json
import matplotlib.pyplot as plt


def plot_classification_training_result():
    base_path = "federated_fall_detection/result/classification"
    files = [
        "UpFall_results.json",
        "federated_SiSFall_results.json",
        "SiSFall_results.json"
    ]
    for file_path in files:
        # Load JSON data from a file
        with open(f"{base_path}/{file_path}", 'r') as file:
            data = json.load(file)

        # Extract data
        epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
        train_loss = [entry["train_loss"] for entry in data["centralized_evaluate"]]
        test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]
        train_accuracy = [entry["train_accuracy"] for entry in data["centralized_evaluate"]]
        test_accuracy = [entry["test_accuracy"] for entry in data["centralized_evaluate"]]

        # Calculate Precision, Recall, and F1 Score
        precision_list = []
        recall_list = []
        f1_score_list = []

        for entry in data["centralized_evaluate"]:
            conf_matrix = entry["test_confution_matrix"]
            TN, FP = conf_matrix[0]
            FN, TP = conf_matrix[1]

            # Calculate precision, recall, and F1
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

        # Plotting Train and Test Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, test_loss, label='Test Loss')
        plt.title('Train and Test Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{base_path}/{file_path[:-13]}_loss.png")

        # Plotting Train and Test Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_accuracy, label='Train Accuracy')
        plt.plot(epochs, test_accuracy, label='Test Accuracy')
        plt.title('Train and Test Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{base_path}/{file_path[:-13]}_accuracy.png")

        # Plotting Precision, Recall, and F1 Score
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, precision_list, label='Precision')
        plt.plot(epochs, recall_list, label='Recall')
        plt.plot(epochs, f1_score_list, label='F1 Score')
        plt.title('Precision, Recall, and F1 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{base_path}/{file_path[:-13]}_metrics.png")


def plot_reconstruction_training_result():
    base_path = "federated_fall_detection/result/centeralized"
    files = [
        "UpFall_results.json",
        "SiSFall_results.json"
    ]
    for file_path in files:
        # Load JSON data from a file
        with open(f"{base_path}/{file_path}", 'r') as file:
            data = json.load(file)

        # Extract data
        epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
        train_loss = [entry["train_loss"] for entry in data["centralized_evaluate"]]
        test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, test_loss, label='Test Loss')

        # Adding title and labels
        plt.title('Train and Test Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.savefig(f"{base_path}/{file_path[:-5]}.png")


def main():
    plot_reconstruction_training_result()
    plot_classification_training_result()


if __name__ == '__main__':
    main()
