# file: create_electrode_csv.py (修正版)

import mne
import pandas as pd
import numpy as np

# 测试
import pprint

print("正在生成64通道电极坐标文件...")

# 1. 加载一个包含我们所需通道的完整标准电极帽配置
try:
    # 从MNE库加载一个名为“standard_1020”的标准电极帽，10-20系统是国际公认的，标准化的头皮上放置电极的系统，确保了
    # 不同研究，不同设备采集的EEG数据在电极位置上的可比性
    # full_montage包含了这个系统上百个电极的名称和他们对应的三维坐标
    full_montage = mne.channels.make_standard_montage("standard_1020")
except Exception as e:
    print(f"无法加载标准电极配置，请检查您的MNE安装是否完整。错误: {e}")
    exit()

# 2. 定义我们项目实际需要的64个通道的名称列表
ch_names_64 = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FT8",
    "T7",
    "T8",
    "T9",
    "T10",
    "TP7",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
    "Iz",
]

# 3. 从完整的电极配置中，筛选出我们需要的64个通道及其对应的3D坐标
#    get_positions()返回的是一个字典，键是通道名，值是坐标
all_positions = full_montage.get_positions()["ch_pos"]
# 查看数据的基本内容
pprint.pprint(full_montage.get_positions())
print("________________________________")
pprint.pprint(all_positions)
positions_64_dict = {ch: all_positions[ch] for ch in ch_names_64 if ch in all_positions}

# 检查是否成功找到了所有64个通道
if len(positions_64_dict) != len(ch_names_64):
    print("警告：并非所有64个通道都在标准配置中找到。")
    print(f"找到 {len(positions_64_dict)} 个, 需要 {len(ch_names_64)} 个。")
    # 找出缺失的通道
    missing_channels = set(ch_names_64) - set(positions_64_dict.keys())
    print(f"缺失的通道: {list(missing_channels)}")


# 4. 使用筛选后的坐标字典创建一个新的、干净的、仅包含64个通道的电极帽对象
montage_64_ch = mne.channels.make_dig_montage(ch_pos=positions_64_dict)

# 5. 从这个新的、状态一致的对象中获取坐标
final_positions = montage_64_ch.get_positions()["ch_pos"]
coords = np.array([final_positions[ch] for ch in ch_names_64 if ch in final_positions])

# 6. 将坐标转换为DataFrame并保存为CSV文件 (单位从米转换为毫米)
df = pd.DataFrame(coords * 1000, columns=["x(mm)", "y(mm)", "z(mm)"])

# 确保 'data' 文件夹存在
import os

if not os.path.exists("data"):
    os.makedirs("data")
output_path = os.path.join("data", "EEG_distance_physionet.csv")
df.to_csv(output_path, index=False)

print(f"文件已成功保存到: {output_path}")
print("文件内容预览:")
print(df.head())
