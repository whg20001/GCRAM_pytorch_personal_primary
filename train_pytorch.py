# file: train_pytorch.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score

# 直接复用原项目中的utils.py
from utils import get_adj, segment_dataset
from model import GCRAM_PyTorch


def main():
    # --- 1. 参数和设备设置 ---
    torch.manual_seed(33)
    np.random.seed(33)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据和模型参数 (与原 train.py 保持一致)
    file_num = 1
    num_node = 64
    model_type = "dg_cram"
    window_size = 400
    step = 10

    # 模型超参数
    kernel_height_1st = 64
    kernel_width_1st = 45
    conv_channel_num = 40
    pooling_width_1st = 75
    pooling_stride_1st = 10
    n_hidden_state = 64

    # 训练超参数
    learning_rate = 1e-5
    training_epochs = 110
    batch_size = 10
    dropout_prob = 0.5
    num_labels = 2

    # --- 2. 数据加载与预处理 ---
    print("正在加载和预处理数据...")
    data = sio.loadmat(f"./data/cross_subject_data_{str(file_num)}.mat")
    test_X = data["test_x"]
    train_X = data["train_x"]
    # PyTorch的CrossEntropyLoss需要类别索引，而不是one-hot编码
    train_y = data["train_y"].ravel().astype(np.int64)
    test_y = data["test_y"].ravel().astype(np.int64)

    adj = get_adj(num_node, model_type.split("_")[0])

    train_X = np.matmul(np.expand_dims(adj, 0), train_X)
    test_X = np.matmul(np.expand_dims(adj, 0), test_X)

    train_raw_x = np.transpose(train_X, [0, 2, 1])
    test_raw_x = np.transpose(test_X, [0, 2, 1])

    train_win_x = segment_dataset(train_raw_x, window_size, step)
    test_win_x = segment_dataset(test_raw_x, window_size, step)

    # 调整维度以匹配PyTorch Conv2d的输入 (N, C, H, W)
    # 原TF输入: [trial, window, time_length, channel]
    # 目标PyTorch输入: [trial, window, channel_in, time_length, channel_node]
    features_train = np.transpose(train_win_x, [0, 1, 3, 2])
    features_test = np.transpose(test_win_x, [0, 1, 3, 2])

    # 增加通道维度
    features_train = np.expand_dims(features_train, axis=2)
    features_test = np.expand_dims(features_test, axis=2)
    num_timestep = features_train.shape[1]

    print("训练特征 shape:", features_train.shape)  # (n_samples, n_windows, C, H, W)
    print("测试特征 shape:", features_test.shape)

    # --- 3. 创建PyTorch DataLoader ---
    train_dataset = TensorDataset(
        torch.from_numpy(features_train).float(), torch.from_numpy(train_y).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(features_test).float(), torch.from_numpy(test_y).long()
    )

    # drop_last=True 确保所有批次大小一致，避免维度错误
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # --- 4. 初始化模型、损失函数和优化器 ---
    model = GCRAM_PyTorch(
        num_labels=num_labels,
        num_node=num_node,
        conv_channel_num=conv_channel_num,
        kernel_height=kernel_height_1st,
        kernel_width=kernel_width_1st,
        pool_width=pooling_width_1st,
        pool_stride=pooling_stride_1st,
        n_hidden_state=n_hidden_state,
        dropout_prob=dropout_prob,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. 训练与评估循环 ---
    for epoch in range(training_epochs):
        # 训练阶段
        model.train()  # 设置为训练模式
        train_loss, train_correct, train_total = 0, 0, 0

        for i, (batch_x, batch_y) in enumerate(train_loader):
            b_size = batch_x.shape[0]
            # Reshape for CNN: (N, T, C, H, W) -> (N*T, C, H, W)
            # H=num_node, W=window_size
            batch_x = batch_x.view(-1, 1, num_node, window_size).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, b_size, num_timestep)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        print(
            f"({time.asctime()}) Epoch: {epoch+1:03d} | 训练损失: {avg_train_loss:.4f} | 训练精度: {train_accuracy:.4f}"
        )

        # 评估阶段
        model.eval()  # 设置为评估模式
        test_loss, test_correct, test_total = 0, 0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():  # 在评估期间不计算梯度
            for batch_x, batch_y in test_loader:
                b_size = batch_x.shape[0]
                batch_x = batch_x.view(-1, 1, num_node, window_size).to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x, b_size, num_timestep)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

                # 为计算AUC保存标签和概率
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total

        try:
            auc_roc_test = roc_auc_score(
                np.array(all_labels), np.array(all_probs)[:, 1]
            )
            print(
                f"({time.asctime()}) Epoch: {epoch+1:03d} | 测试AUC: {auc_roc_test:.4f} | 测试损失: {avg_test_loss:.4f} | 测试精度: {test_accuracy:.4f}\n"
            )
        except ValueError:
            print(
                f"({time.asctime()}) Epoch: {epoch+1:03d} | 测试损失: {avg_test_loss:.4f} | 测试精度: {test_accuracy:.4f} (AUC无法计算)\n"
            )


if __name__ == "__main__":
    main()
