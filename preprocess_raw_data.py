# file: preprocess_raw_data.py

import mne
import numpy as np
import scipy.io
import os
from tqdm import tqdm
import random


def preprocess_physionet_data(raw_data_path, output_mat_path):
    """
    处理PhysioNet EEG Motor Movement/Imagery Dataset的原始数据。
    """
    print("开始预处理原始EEG数据...")

    all_subjects_data = {}

    # [cite_start]论文中提到移除了4位被试 [cite: 333]
    subjects_to_exclude = {88, 89, 92, 100}

    # 定义运动想象任务对应的运行(run)编号
    # 根据数据集描述：
    # 01, 02: 基线，睁眼/闭眼
    # 03, 07, 11: 想象左/右拳开合
    # 04, 08, 12: 想象双脚/双手开合
    # 05, 09, 13: 实际左/右拳开合
    # 06, 10, 14: 实际双脚/双手开合
    # 论文中只使用了左/右拳运动想象，所以我们只关心 run 03, 07, 11
    imagery_runs = ["03", "07", "11"]

    # 事件ID与标签的映射
    # T0: rest, T1: a imagem de abertura e fecho do punho esquerdo, T2: a imagem de abertura e fecho do punho direito
    event_id = {"T1": 1, "T2": 2}  # 1: 左拳, 2: 右拳

    # 遍历所有被试文件夹 (S001, S002, ...)
    subject_dirs = [
        d
        for d in os.listdir(raw_data_path)
        if d.startswith("S") and os.path.isdir(os.path.join(raw_data_path, d))
    ]

    for subject_dir in tqdm(subject_dirs, desc="处理被试数据"):
        subject_id = int(subject_dir[1:])
        if subject_id in subjects_to_exclude:
            continue

        subject_epochs = []
        subject_labels = []

        for run in imagery_runs:
            file_name = f"S{str(subject_id).zfill(3)}R{run}.edf"
            file_path = os.path.join(raw_data_path, subject_dir, file_name)

            if not os.path.exists(file_path):
                continue

            # 使用MNE读取EDF文件
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # 从文件名中提取事件
            events, _ = mne.events_from_annotations(
                raw, event_id=event_id, verbose=False
            )

            # 提取Epochs (试验片段)
            # [cite_start]论文提到试验持续约3.1秒，采样率160Hz [cite: 331, 332]
            # 我们从事件开始时提取约3.1秒的数据
            tmin, tmax = (
                0.0,
                3.1 - 1 / raw.info["sfreq"],
            )  # 减去一个采样点的时间确保长度正确

            # [cite_start]baseline=None表示不进行基线校正，与论文中直接使用原始数据一致 [cite: 343]
            epochs = mne.Epochs(
                raw,
                events,
                event_id=list(event_id.values()),
                tmin=tmin,
                tmax=tmax,
                proj=False,
                baseline=None,
                preload=True,
                verbose=False,
            )

            # 将标签从1,2转换为0,1
            labels = epochs.events[:, -1] - 1

            subject_epochs.append(epochs.get_data(picks="eeg"))
            subject_labels.append(labels)

        if subject_epochs:
            all_subjects_data[subject_id] = {
                "data": np.concatenate(subject_epochs, axis=0),
                "labels": np.concatenate(subject_labels, axis=0),
            }

    # --- 按被试划分训练集和测试集 ---
    print("正在划分训练集和测试集...")
    all_subject_ids = list(all_subjects_data.keys())
    random.shuffle(all_subject_ids)

    # [cite_start]论文中使用了9个交叉验证集，每个集有10个测试被试 [cite: 334]
    # 这里我们只创建一个固定的划分，选择10个被试作为测试集
    test_subject_ids = all_subject_ids[:10]
    train_subject_ids = all_subject_ids[10:]

    print(f"测试集被试ID: {sorted(test_subject_ids)}")
    print(f"训练集被试ID数量: {len(train_subject_ids)}")

    train_x, train_y = [], []
    for sid in train_subject_ids:
        train_x.append(all_subjects_data[sid]["data"])
        train_y.append(all_subjects_data[sid]["labels"])

    test_x, test_y = [], []
    for sid in test_subject_ids:
        test_x.append(all_subjects_data[sid]["data"])
        test_y.append(all_subjects_data[sid]["labels"])

    # 合并数据并调整维度以匹配项目要求 (样本数, 通道数, 时间点数)
    train_x_np = np.concatenate(train_x, axis=0)
    train_y_np = np.concatenate(train_y, axis=0).reshape(-1, 1)
    test_x_np = np.concatenate(test_x, axis=0)
    test_y_np = np.concatenate(test_y, axis=0).reshape(-1, 1)

    # --- 保存为 .mat 文件 ---
    mat_dict = {
        "train_x": train_x_np,
        "train_y": train_y_np,
        "test_x": test_x_np,
        "test_y": test_y_np,
    }

    scipy.io.savemat(output_mat_path, mat_dict)
    print(f"预处理完成！数据已保存至: {output_mat_path}")
    print("数据维度:")
    print(f"train_x: {train_x_np.shape}")
    print(f"train_y: {train_y_np.shape}")
    print(f"test_x: {test_x_np.shape}")
    print(f"test_y: {test_y_np.shape}")


if __name__ == "__main__":
    # !!!重要!!! 请将此路径修改为您下载和解压后的 'files' 文件夹的实际路径
    RAW_DATA_FOLDER_PATH = "data/files"

    # 确保'data'文件夹存在
    if not os.path.exists("data"):
        os.makedirs("data")
    OUTPUT_MAT_FILE_PATH = os.path.join("data", "cross_subject_data_1.mat")

    preprocess_physionet_data(RAW_DATA_FOLDER_PATH, OUTPUT_MAT_FILE_PATH)
