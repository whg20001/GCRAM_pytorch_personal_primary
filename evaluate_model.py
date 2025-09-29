import torch
import numpy as np
import scipy.io as sio
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 从项目文件中导入必要的模块
from model import GCRAM_PyTorch
from utils import get_adj, segment_dataset


def evaluate(args):
    """主评估函数"""
    # --- 1. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 定义模型架构 ---
    # 注意：这里的参数必须与您训练时使用的模型参数完全一致！
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

    # --- 3. 加载已保存的模型权重 ---
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件未找到于路径 {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()  # 非常重要：将模型设置为评估模式！
    print(f"模型权重已从 {args.model_path} 加载。")

    # --- 4. 加载并预处理测试数据 (内存密集型) ---
    print("正在加载和预处理测试数据...")
    data = sio.loadmat(args.data_path)
    test_X_raw = data["test_x"]
    test_y = data["test_y"].ravel().astype(np.int64)

    adj = get_adj(
        args.num_node, args.graph_type, data_dir=os.path.dirname(args.data_path)
    )
    test_X_embedded = np.matmul(np.expand_dims(adj, 0), test_X_raw)
    test_raw_x = np.transpose(test_X_embedded, [0, 2, 1])

    print("正在为测试集生成所有滑动窗口 (此步骤可能需要大量内存)...")
    test_win_x = segment_dataset(test_raw_x, args.window_size, args.step)

    features_test = np.transpose(test_win_x, [0, 1, 3, 2])
    features_test = np.expand_dims(features_test, axis=2)
    num_timestep = features_test.shape[1]

    test_dataset = TensorDataset(
        torch.from_numpy(features_test).float(), torch.from_numpy(test_y).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 5. 执行推理并收集结果 ---
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="正在评估"):
            b_size = batch_x.shape[0]
            batch_x = batch_x.view(-1, 1, args.num_node, args.window_size).to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, b_size, num_timestep)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- 6. 计算并打印评估指标 ---
    print("\n--- 模型评估结果 ---")
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    print(f"整体准确率 (Accuracy): {accuracy:.4f}")
    print(f"ROC-AUC 分数: {roc_auc:.4f}")
    print("\n分类报告:")
    print(
        classification_report(
            all_labels, all_preds, target_names=["类别 0 (左拳)", "类别 1 (右拳)"]
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["预测为 0", "预测为 1"],
        yticklabels=["实际为 0", "实际为 1"],
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-CRAM PyTorch Model Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/gcram_pytorch_model.pth",
        help="已保存模型文件的路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/cross_subject_data_1.mat",
        help="数据文件的路径",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="dg",
        choices=["ng", "dg", "sg"],
        help="图类型，必须与被评估的模型一致",
    )
    parser.add_argument("--num_node", type=int, default=64, help="EEG节点数")
    parser.add_argument("--window_size", type=int, default=400, help="滑动窗口大小")
    parser.add_argument("--step", type=int, default=10, help="滑动窗口步长")
    parser.add_argument("--batch_size", type=int, default=10, help="评估时的批次大小")

    args = parser.parse_args()
    evaluate(args)
