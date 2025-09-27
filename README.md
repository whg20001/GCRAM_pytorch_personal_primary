# README

1. 数据集地址：https://physionet.org/content/eegmmidb/get-zip/1.0.0/

    - 直接输入到迅雷下载即可（迅雷下载速度比较快）

2. 本项目是基于GCRAM修改而来，原项目地址：https://github.com/dalinzhang/GCRAM

3. 修改原由：

    - 原项目是通过Tensorflow编写，但是本人所学框架是pytorch，所以进行了改写
    - 原项目的README内容不是很完善，对初学者不是很友好

4. 新的文件结构如下：

    [![pVI4xHK.png](https://s21.ax1x.com/2025/09/27/pVI4xHK.png)](https://imgse.com/i/pVI4xHK)

5. 项目复现步骤及环境配置：

    - 使用的pytorch版本是11.4
    - 第一步：在根目录下新建data文件夹，然后将数据集的file文件夹复制到data文件夹即可
    - 第二步：通过create_electrode_csv.py文件创建电极帽的空间信息csv文件
    - 第三步：通过preprocess_raw_data.py文件创建基础的设置（读取所有109位被试的原始`.edf`数据文件和`.event`事件文件，提取出与“左/右拳运动想象”相关的脑电信号片段（这称为Epoching），然后按照论文的设定划分训练集和测试集，最后保存为单一的`.mat`文件。）
    - 第四步：运行train_pytorch.py文件进行训练
    - 