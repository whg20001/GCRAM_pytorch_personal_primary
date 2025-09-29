import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.io as sio
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# 导入我们的模块化代码
from model import GCRAM_PyTorch
from utils import get_adj, segment_dataset


def train(args):
    """主训练函数"""
    # --- 1. 设置 ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 数据加载与预处理 (内存密集型) ---
    print("正在加载和预处理数据...")
    data_path = os.path.join(args.data_dir, f"cross_subject_data_{args.file_num}.mat")
    data = sio.loadmat(data_path)
    train_X_raw, test_X_raw = data["train_x"], data["test_x"]
    train_y, test_y = data["train_y"].ravel().astype(np.int64), data[
        "test_y"
    ].ravel().astype(np.int64)

    adj = get_adj(args.num_node, args.graph_type, data_dir=args.data_dir)
    train_X_embedded = np.matmul(np.expand_dims(adj, 0), train_X_raw)
    test_X_embedded = np.matmul(np.expand_dims(adj, 0), test_X_raw)

    train_raw_x = np.transpose(train_X_embedded, [0, 2, 1])
    test_raw_x = np.transpose(test_X_embedded, [0, 2, 1])

    print("正在生成所有滑动窗口 (此步骤可能需要大量内存)...")
    train_win_x = segment_dataset(train_raw_x, args.window_size, args.step)
    test_win_x = segment_dataset(test_raw_x, args.window_size, args.step)

    # 调整维度以匹配PyTorch Conv2d的输入 (N, T, C, H, W)
    features_train = np.transpose(train_win_x, [0, 1, 3, 2])
    features_test = np.transpose(test_win_x, [0, 1, 3, 2])
    features_train = np.expand_dims(features_train, axis=2)
    features_test = np.expand_dims(features_test, axis=2)
    num_timestep = features_train.shape[1]

    print("训练特征 shape:", features_train.shape)
    print("测试特征 shape:", features_test.shape)

    # --- 3. 创建DataLoader ---
    train_dataset = TensorDataset(
        torch.from_numpy(features_train).float(), torch.from_numpy(train_y).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(features_test).float(), torch.from_numpy(test_y).long()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    # --- 4. 初始化模型、损失函数和优化器 ---
    model = GCRAM_PyTorch(
        num_labels=2,
        num_node=args.num_node,
        conv_channel_num=40,
        kernel_height=64,
        kernel_width=45,
        pool_width=75,
        pool_stride=10,
        n_hidden_state=64,
        dropout_prob=0.5,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 5. 训练与评估循环 ---
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True
        )
        for batch_x, batch_y in progress_bar:
            b_size = batch_x.shape[0]
            # Reshape for CNN: (N, T, C, H, W) -> (N*T, C, H, W)
            batch_x = batch_x.view(-1, 1, args.num_node, args.window_size).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x, b_size, num_timestep)
            loss = criterion(outputs, batch_y)

            # --- 应用正则化 ---
            if args.regularization == "l1":
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += args.lambda_reg * l1_penalty
            elif args.regularization == "group_lasso":
                group_lasso_penalty = 0
                for kernel in model.conv_block[0].parameters():
                    if kernel.dim() > 1:
                        group_lasso_penalty += torch.sqrt(
                            torch.sum(kernel**2, dim=[1, 2, 3])
                        ).sum()
                loss += args.lambda_reg * group_lasso_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- 每个Epoch后进行评估 ---
        model.eval()
        test_correct, test_total = 0, 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                b_size = batch_x.shape[0]
                batch_x = batch_x.view(-1, 1, args.num_node, args.window_size).to(
                    device
                )
                batch_y = batch_y.to(device)
                outputs = model(batch_x, b_size, num_timestep)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

        test_accuracy = test_correct / test_total
        auc_roc_test = roc_auc_score(np.array(all_labels), np.array(all_probs)[:, 1])

        print(
            f"({time.asctime()}) Epoch: {epoch+1:03d} | 训练损失: {avg_train_loss:.4f} | 测试精度: {test_accuracy:.4f} | 测试AUC: {auc_roc_test:.4f}"
        )

    # --- 6. 保存模型 ---
    if (epoch + 1) % 20 == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir, f"{args.graph_type}_cram_model_final_LASSO.pth"
        )
        torch.save(model.state_dict(), save_path)
        print(f"训练完成，模型已保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-CRAM PyTorch Model Training")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="数据文件所在目录"
    )
    parser.add_argument("--file_num", type=int, default=1, help="数据文件名编号")
    parser.add_argument(
        "--save_dir", type=str, default="./saved_models", help="模型保存目录"
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="dg",
        choices=["ng", "dg", "sg"],
        help="图类型 (ng, dg, sg)",
    )
    parser.add_argument("--num_node", type=int, default=64, help="EEG节点数")
    parser.add_argument("--window_size", type=int, default=400, help="滑动窗口大小")
    parser.add_argument("--step", type=int, default=10, help="滑动窗口步长")
    parser.add_argument("--epochs", type=int, default=120, help="训练周期数")
    parser.add_argument(
        "--batch_size", type=int, default=10, help="批次大小 (试验的数量)"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=33, help="随机种子")
    parser.add_argument(
        "--regularization",
        type=str,
        default="none",
        choices=["none", "l1", "group_lasso"],
        help="正则化类型",
    )
    parser.add_argument("--lambda_reg", type=float, default=1e-5, help="正则化系数")

    args = parser.parse_args()
    train(args)
