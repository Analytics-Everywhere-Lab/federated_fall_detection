import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_path = "/home/s7wu7/project/federated_fall_detection/datasets"
raw_file_base_path = f"{base_path}/raw"


def read_meta():
    f = open(f'{base_path}/cleaned_meta.json')
    data = json.load(f)
    f.close()
    return data


def save_meta(meta_json):
    f = open(f'{base_path}/cleaned_meta.json', "w")
    json.dump(meta_json, f, ensure_ascii=False, indent=4)
    f.close()


def read_labels():
    df = pd.read_csv(f"{base_path}/activity_map.csv")
    return df


def clean_up_fall(with_gyro):
    df = pd.read_csv(f"{raw_file_base_path}/Up-Fall/CompleteDataSet.csv")
    belt_columns = df.columns[0:1].tolist() + df.columns[15:21].tolist() + df.columns[-4:].tolist()
    df = df[belt_columns].drop(0)
    df.columns = ['TimeStamps', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Subject', 'Activity', 'Trial',
                  'Tag']
    df["TimeStamps"] = pd.to_datetime(df["TimeStamps"])
    new_dtype = {'acc_x': "float16", 'acc_y': "float16", 'acc_z': "float16", 'gyro_x': "float16", 'gyro_y': "float16",
                 'gyro_z': "float16", 'Subject': "int8", 'Activity': "int8", 'Trial': "int8", 'Tag': "int8"}
    df = df.astype(new_dtype)
    dataset_name = "UpFall"
    meta_json = read_meta()
    this_meta = []

    for i in range(len(meta_json["files"]) - 1, -1, -1):
        file = meta_json["files"][i]
        if file['dataset'] == dataset_name:
            del meta_json["files"][i]
    for f in glob.glob(f"{base_path}/cleaned/{dataset_name}_*.csv"):
        os.remove(f)

    labels = read_labels()

    def save_single_file(temp_df: pd.DataFrame):
        activity = temp_df["Activity"].values[0]
        subject = temp_df["Subject"].values[0]
        trial = temp_df["Trial"].values[0]
        temp_df["TimeStamps"] = temp_df["TimeStamps"] - temp_df["TimeStamps"].min()
        temp_df["TimeStamps"] = temp_df["TimeStamps"].dt.total_seconds() * 1000
        temp_new_dtype = {"TimeStamps": "int32"}
        temp_df = temp_df.astype(temp_new_dtype)
        if with_gyro:
            temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z", 'gyro_x', 'gyro_y', 'gyro_z']]
            temp_df.columns = ["TimeStamps", "X", "Y", "Z", 'GyrX', 'GyrY', 'GyrZ']
        else:
            temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z"]]
            temp_df.columns = ["TimeStamps", "X", "Y", "Z"]
        label_name = labels[labels["UP FALL"] == activity]["ACTIVITY"].values[0]
        is_fall = labels[labels["UP FALL"] == activity]["IS FALL"].values[0]
        label = labels[labels["UP FALL"] == activity]["LABEL"].values[0]
        file_name = f"{dataset_name}_{subject}_{label_name}_{trial}.csv"

        this_meta.append({
            "file_name": str(file_name),
            "activity": "Fall" if is_fall else "ADL",
            "subject": int(subject),
            "trial": int(trial),
            "dataset_label": int(activity),
            "label": int(label),
            "label_name": str(labels.iloc[label, 0]),
            "dataset": dataset_name
        })
        temp_df.to_csv(f"{base_path}/cleaned/{file_name}", index=False)
        print(file_name)

    df.groupby(["Activity", "Trial", "Subject"]).apply(save_single_file)
    meta_json['files'] += this_meta
    save_meta(meta_json)


def clean_sis_fall(with_gyro):
    path = f"{raw_file_base_path}/SisFall/SisFall_dataset/"
    folders = os.listdir(path)
    acc1_coe = ((2 * 16) / (2 ** 13))
    acc2_coe = ((2 * 8) / (2 ** 14))
    gyro_coe = ((2 * 2000) / (2 ** 16))

    dataset_name = "SiSFall"
    labels = read_labels()
    meta_json = read_meta()

    for i in range(len(meta_json["files"]) - 1, -1, -1):
        file = meta_json["files"][i]
        if file['dataset'] == dataset_name:
            del meta_json["files"][i]
    for f in glob.glob(f"{base_path}/cleaned/{dataset_name}_*.csv"):
        os.remove(f)

    this_meta = []
    for folder in folders:
        if os.path.isdir(f"{path}/{folder}"):
            files = os.listdir(f"{path}/{folder}")
            for file in files:
                if file.endswith(".txt"):
                    parts = file[:-4].split("_")
                    temp_df = pd.read_csv(f"{path}/{folder}/{file}", header=None)
                    temp_df.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc2_x', 'acc2_y',
                                       'acc2_z']
                    temp_df['acc2_z'] = temp_df['acc2_z'].apply(lambda x: int(str(x)[:-1]))
                    temp_df[['acc_x', 'acc_y', 'acc_z']] = temp_df[['acc_x', 'acc_y', 'acc_z']] * acc1_coe
                    temp_df[['acc2_x', 'acc2_y', 'acc2_z']] = temp_df[['acc2_x', 'acc2_y', 'acc2_z']] * acc2_coe
                    temp_df[['gyro_x', 'gyro_y', 'gyro_z']] = temp_df[['gyro_x', 'gyro_y', 'gyro_z']] * gyro_coe
                    temp_df["TimeStamps"] = pd.date_range(start="2017-01-01", periods=temp_df.shape[0], freq="5ms")
                    temp_df["TimeStamps"] = temp_df["TimeStamps"] - temp_df["TimeStamps"].min()
                    temp_df["TimeStamps"] = temp_df["TimeStamps"].dt.total_seconds() * 1000
                    temp_new_dtype = {"TimeStamps": "int16"}
                    temp_df = temp_df.astype(temp_new_dtype)
                    if with_gyro:
                        temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z", 'gyro_x', 'gyro_y', 'gyro_z']]
                        temp_df.columns = ["TimeStamps", "X", "Y", "Z", 'GyrX', 'GyrY', 'GyrZ']
                    else:
                        temp_df = temp_df[["TimeStamps", "acc_x", "acc_y", "acc_z"]]
                        temp_df.columns = ["TimeStamps", "X", "Y", "Z"]

                    label_name = int(parts[0][1:])
                    if parts[0].startswith("D"):
                        label_name += 15

                    subject = int(parts[1][2:])
                    if parts[1].startswith("SE"):
                        subject += 23
                    is_fall = labels[labels["SIS FALL"] == label_name]["IS FALL"].values[0]
                    label = labels[labels["SIS FALL"] == label_name]["LABEL"].values[0]
                    activity = label_name
                    label_name = labels[labels["SIS FALL"] == label_name]["ACTIVITY"].values[0]
                    trial = int(parts[2][1:])

                    # saving
                    file_name = f"{dataset_name}_{subject}_{label_name}_{trial}.csv"
                    this_meta.append({
                        "file_name": str(file_name),
                        "activity": "Fall" if is_fall else "ADL",
                        "subject": subject,
                        "trial": trial,
                        "dataset_label": int(activity),
                        "label": int(label),
                        "label_name": str(label_name),
                        "dataset": dataset_name
                    })

                    temp_df.to_csv(f"{base_path}/cleaned/{file_name}", index=False)
                    print(file_name)

    meta_json['files'] += this_meta
    save_meta(meta_json)


def clean_datasets(with_gyro):
    clean_up_fall(with_gyro)
    clean_sis_fall(with_gyro)


def read_datasets(frequency, dataset_name):
    meta_json = read_meta()
    datasets = []
    labels = []
    files = []
    for i in range(len(meta_json["files"])):
        file = meta_json["files"][i]
        if file["dataset"] == dataset_name:
            file_name = file["file_name"]
            temp_df = pd.read_csv(f"{base_path}/cleaned/" + file_name)
            temp_df["TimeStamps"] = pd.to_datetime(temp_df["TimeStamps"], unit="ms")
            temp_df = temp_df.set_index("TimeStamps")
            try:
                up_sample = temp_df.resample("ms").interpolate("linear")
            except ValueError:
                temp_df = temp_df[~temp_df.index.duplicated(keep="first")]
                up_sample = temp_df.resample("ms").interpolate("linear")
            temp_df = up_sample.resample(frequency).interpolate("linear")
            # temp_df = temp_df[['X']]
            labels.append(file["dataset_label"])
            datasets.append(temp_df)
            files.append(file)
    return datasets, labels, files


def load_data(dataset_name="UpFall", frequancy="50ms", window_size=40, window_step=20, user_split=True,
              two_class=False, scaling=False, normlize=False, reload=False):
    saving_path = f"{base_path}/saved"
    if reload:
        test_dataset = np.load(f"{saving_path}/test_dataset.npy")
        test_labels = np.load(f"{saving_path}/test_labels.npy")
        train_dataset = np.load(f"{saving_path}/train_dataset.npy")
        train_labels = np.load(f"{saving_path}/train_labels.npy")
        if user_split:
            train_users = np.load(f"{saving_path}/train_users.npy")
            test_users = np.load(f"{saving_path}/test_users.npy")
            return train_dataset, test_dataset, train_labels, test_labels, train_users, test_users
        return train_dataset, test_dataset, train_labels, test_labels

    raw_datasets, raw_label, raw_files = read_datasets(frequancy, dataset_name)

    def create_sequences(temp_df):
        temp_sample_count = ((temp_df.shape[0] - window_size) // window_step) + 1
        output = []
        temp_df = temp_df.to_numpy()
        for i in range(temp_sample_count):
            start_index = i * window_step
            end_index = start_index + window_size
            output.append(temp_df[start_index:end_index])
        return np.stack(output)

    def build_dataset(datasets, labels, files, inner_frequency):
        final_dataset = []
        final_labels = []
        final_users = []
        final_classes = []
        inner_frequency = int(inner_frequency[:2])
        for j in range(len(datasets)):
            if j % 100 == 0:
                print(f"LOAD DATASET: {j / len(datasets)}")
            file = files[j]
            data = datasets[j]
            input_data = data.dropna()
            input_data = input_data.ewm(span=5, adjust=False).mean()
            if input_data.shape[0] <= window_size:
                continue
            windows_df = create_sequences(input_data)
            if np.isnan(windows_df).sum() > 0:
                windows_df = windows_df.dropna()

            if two_class:
                label = np.zeros(windows_df.shape[0])
                if file["activity"] == "Fall":
                    momentom = np.abs(windows_df).sum(axis=1)
                    acc_momentom = momentom[:, 3:].mean(axis=1)
                    gyro_momentom = momentom[:, :3].mean(axis=1)
                    highest_acc_momentom = acc_momentom > acc_momentom.min() + (
                            7 * (acc_momentom.max() - acc_momentom.min()) / 10)
                    highest_gyro_momentom = gyro_momentom > gyro_momentom.min() + (
                            7 * (gyro_momentom.max() - gyro_momentom.min()) / 10)
                    label[highest_gyro_momentom * highest_acc_momentom] = 1

                    windows_df = windows_df[label == 1]
                    label = label[label == 1]
            else:
                if file["activity"] == "Fall":
                    label = np.zeros(windows_df.shape[0])
                    momentom = np.abs(windows_df).sum(axis=1)
                    acc_momentom = momentom[:, 3:].mean(axis=1)
                    gyro_momentom = momentom[:, :3].mean(axis=1)
                    highest_acc_momentom = acc_momentom > acc_momentom.min() + (
                            7 * (acc_momentom.max() - acc_momentom.min()) / 10)
                    highest_gyro_momentom = gyro_momentom > gyro_momentom.min() + (
                            7 * (gyro_momentom.max() - gyro_momentom.min()) / 10)
                    label[highest_gyro_momentom * highest_acc_momentom] = labels[j]

                    windows_df = windows_df[label == labels[j]]
                    label = label[label == labels[j]]
                else:
                    label = np.full(windows_df.shape[0], labels[j])

            final_dataset.append(windows_df)
            final_labels.append(label)
            final_users.append(np.array([file['subject']] * windows_df.shape[0]))
            final_classes.append(np.full(windows_df.shape[0], labels[j]))

        final_dataset = np.concatenate(final_dataset)
        final_labels = np.concatenate(final_labels)
        final_classes = np.concatenate(final_classes)
        final_users = np.concatenate(final_users)
        return final_dataset, final_labels, final_classes, final_users

    windows_datasets, labels_datasets, class_datasets, final_users = build_dataset(
        raw_datasets, raw_label, raw_files, frequancy
    )

    if user_split:
        train_users, test_users = train_test_split(np.unique(final_users), test_size=0.4, random_state=42)
        train_users_index = np.isin(final_users, train_users)
        test_users_index = np.isin(final_users, test_users)

        train_dataset = windows_datasets[train_users_index]
        test_dataset = windows_datasets[test_users_index]
        train_labels = labels_datasets[train_users_index]
        test_labels = labels_datasets[test_users_index]
        train_users = final_users[train_users_index]
        test_users = final_users[test_users_index]
        p = np.random.permutation(train_users.shape[0])
        train_dataset, train_labels, train_users = train_dataset[p], train_labels[p], train_users[p]
        p = np.random.permutation(test_users.shape[0])
        test_dataset, test_labels, test_users = test_dataset[p], test_labels[p], test_users[p]
    else:
        train_dataset, test_dataset, train_labels, test_labels = train_test_split(
            windows_datasets, labels_datasets, test_size=0.3, random_state=42
        )

    if scaling:
        scaler = StandardScaler()
        train_shape = train_dataset.shape
        test_shape = test_dataset.shape
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1, train_shape[-1])).reshape(train_shape)
        test_dataset = scaler.transform(test_dataset.reshape(-1, test_shape[-1])).reshape(test_shape)

    if normlize:
        scaler = MinMaxScaler((-1, 1))
        train_shape = train_dataset.shape
        test_shape = test_dataset.shape
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1, train_shape[-1])).reshape(train_shape)
        test_dataset = scaler.transform(test_dataset.reshape(-1, test_shape[-1])).reshape(test_shape)

    test_dataset = test_dataset.astype("float32")
    test_labels = test_labels.astype("float32")
    train_dataset = train_dataset.astype("float32")
    train_labels = train_labels.astype("float32")

    Path(f"{saving_path}").mkdir(parents=True, exist_ok=True)
    np.save(f"{saving_path}/test_dataset.npy", test_dataset)
    np.save(f"{saving_path}/test_labels.npy", test_labels)
    np.save(f"{saving_path}/train_dataset.npy", train_dataset)
    np.save(f"{saving_path}/train_labels.npy", train_labels)
    if user_split:
        train_users = train_users.astype("float32")
        test_users = test_users.astype("float32")
        np.save(f"{saving_path}/train_users.npy", train_users)
        np.save(f"{saving_path}/test_users.npy", test_users)
        return train_dataset, test_dataset, train_labels, test_labels, train_users, test_users

    return train_dataset, test_dataset, train_labels, test_labels


def main():
    # clean_datasets(True)
    data = load_data(reload=True)
    print()


if __name__ == '__main__':
    main()
