# file: predict.py

import torch
import numpy as np
import os

# 从项目文件中导入必要的模块
from model import GCRAM_PyTorch
from utils import get_adj, segment_signal_without_transition


def predict_single_trial(model, raw_eeg_trial, device, model_params):
    """
    使用加载好的模型对单次EEG试验数据进行预测。

    参数:
    model (torch.nn.Module): 已加载并设置为评估模式的模型。
    raw_eeg_trial (numpy.array): 单次试验的原始EEG数据，
                                 维度应为 (n_channels, n_timesteps)，例如 (64, 497)。
    device (torch.device): 运行模型的设备 (cpu 或 cuda)。
    model_params (dict): 包含模型和预处理所需参数的字典。

    返回:
    predicted_class (int): 预测的类别索引 (0 或 1)。
    class_probabilities (dict): 包含每个类别预测概率的字典。
    """

    # --- 1. 预处理输入数据 ---
    #    这个流程必须与训练和评估时完全一致！

    print("正在预处理单次试验数据...")
    # a. 获取图邻接矩阵
    adj = get_adj(model_params["num_node"], model_params["model_type"].split("_")[0])

    # b. 应用图嵌入
    #    增加一个虚拟的 'batch' 维度，然后进行矩阵乘法，最后移除该维度。
    eeg_graph_embedded = np.matmul(
        np.expand_dims(adj, 0), np.expand_dims(raw_eeg_trial, 0)
    )[0]

    # c. 调整维度以进行滑动窗口处理
    #    输入维度: (n_channels, n_timesteps) -> (n_timesteps, n_channels)
    eeg_transposed = np.transpose(eeg_graph_embedded, (1, 0))

    # d. 应用滑动窗口
    windows = segment_signal_without_transition(
        eeg_transposed, model_params["window_size"], model_params["step"]
    )
    #    输出维度: (n_windows, window_size, n_channels)

    if windows.shape[0] == 0:
        raise ValueError("输入数据的长度不足以生成任何滑动窗口，请检查时间点数量。")

    # e. 调整维度以匹配模型输入
    #    (n_windows, window_size, n_channels) -> (n_windows, 1, n_channels, window_size)
    windows_final = np.transpose(
        windows, (0, 2, 1)
    )  # -> (n_windows, n_channels, window_size)
    windows_final = np.expand_dims(
        windows_final, axis=1
    )  # -> (n_windows, 1, n_channels, window_size)

    # f. 转换为PyTorch Tensor
    input_tensor = torch.from_numpy(windows_final).float().to(device)

    # --- 2. 模型推理 ---
    print("数据已准备好，正在进行模型推理...")
    with torch.no_grad():  # 关闭梯度计算
        # 模型需要知道原始的batch_size（在这里是1个trial）和窗口数量
        outputs = model(input_tensor, batch_size=1, num_timestep=input_tensor.shape[0])

    # --- 3. 处理输出结果 ---
    # a. 计算概率
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()

    # b. 获取预测类别
    predicted_index = torch.argmax(probabilities).item()

    class_probabilities = {
        "类别 0 (左拳)": probabilities[0].item(),
        "类别 1 (右拳)": probabilities[1].item(),
    }

    return predicted_index, class_probabilities


if __name__ == "__main__":
    # --- 1. 配置参数 ---
    # 这些参数必须与您用来训练和评估模型的参数完全一致
    MODEL_PARAMS = {
        "num_node": 64,
        "model_type": "dg_cram",
        "window_size": 400,
        "step": 10,
        # 模型自身的架构参数
        "num_labels": 2,
        "conv_channel_num": 40,
        "kernel_height": 64,
        "kernel_width": 45,
        "pool_width": 75,
        "pool_stride": 10,
        "n_hidden_state": 64,
        "dropout_prob": 0.5,
    }

    SAVED_MODEL_PATH = "./saved_models/gcram_pytorch_model.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载模型 ---
    # a. 初始化模型架构
    model = GCRAM_PyTorch(**MODEL_PARAMS).to(DEVICE)

    # b. 加载训练好的权重
    if not os.path.exists(SAVED_MODEL_PATH):
        raise FileNotFoundError(
            f"错误: 找不到已保存的模型文件于路径 {SAVED_MODEL_PATH}"
        )
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    model.eval()  # **至关重要**: 将模型设置为评估/推理模式
    print(f"模型已从 {SAVED_MODEL_PATH} 加载，并设置为评估模式。")

    # --- 3. 创建一个虚拟的EEG数据样本进行预测 ---
    # 在真实应用中，您会从设备或文件中加载真实的EEG数据
    # 维度: (通道数, 时间点数)
    # 根据PhysioNet数据集，时间点数约为497 (3.1秒 * 160Hz)
    print("\n--- 准备单个预测样本 ---")

    # 创建一个与真实数据维度相似的随机样本
    # np.float64是为了在应用图嵌入时保持精度，模型内部会处理为float32
    sample_eeg_data = np.random.randn(MODEL_PARAMS["num_node"], 497).astype(np.float64)
    print(f"创建了一个虚拟EEG样本，维度为: {sample_eeg_data.shape}")

    # --- 4. 调用预测函数 ---
    try:
        predicted_class, probabilities = predict_single_trial(
            model, sample_eeg_data, DEVICE, MODEL_PARAMS
        )

        # --- 5. 打印结果 ---
        print("\n--- 预测结果 ---")
        print(f"预测类别索引: {predicted_class}")
        predicted_label = "类别 0 (左拳)" if predicted_class == 0 else "类别 1 (右拳)"
        print(f"预测意图: {predicted_label}")
        print("\n详细概率分布:")
        for label, prob in probabilities.items():
            print(f"  - {label}: {prob:.4f} ({prob:.2%})")

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
