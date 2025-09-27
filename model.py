# file: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention机制的PyTorch实现。
    原始思路来源于 "Hierarchical Attention Networks for Document Classification"
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 对应TensorFlow中的w_omega, b_omega, u_omega
        # 使用nn.Linear可以更方便地处理权重和偏置，这里为了与原版更接近，手动创建Parameter
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size))

        self._init_weights()

    def _init_weights(self, stdv=0.1):
        # 权重初始化
        nn.init.normal_(self.w_omega, std=stdv)
        nn.init.normal_(self.b_omega, std=stdv)
        nn.init.normal_(self.u_omega, std=stdv)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_size)
        
        # v = tanh(W_omega * h_t + b_omega)
        # torch.tensordot 功能上等同于 tf.tensordot
        # (B, T, D) * (D, A) -> (B, T, A)
        v = torch.tanh(torch.tensordot(inputs, self.w_omega, dims=([2], [0])) + self.b_omega)
        
        # vu = u_omega^T * v
        # (B, T, A) * (A) -> (B, T)
        vu = torch.tensordot(v, self.u_omega, dims=([2], [0]))
        
        # alphas = softmax(vu)
        alphas = F.softmax(vu, dim=1)
        
        # output = sum(alphas_t * h_t)
        # 使用 unsqueeze 扩展维度以进行广播
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1) # -> (batch_size, hidden_size)
        
        return output, alphas

class GCRAM_PyTorch(nn.Module):
    def __init__(self, num_labels, num_node, conv_channel_num, kernel_height, kernel_width, 
                 pool_width, pool_stride, n_hidden_state, dropout_prob):
        super(GCRAM_PyTorch, self).__init__()
        
        # CNN 部分
        # 将原 cnn_class.py 中的 apply_conv2d 功能用nn.Module实现
        # TensorFlow中的padding='VALID' 等同于 PyTorch中的padding=0
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channel_num,
                kernel_size=(kernel_height, kernel_width),
                stride=1,
                padding=0
            ),
            # tf.layers.batch_normalization 对应 nn.BatchNorm2d
            nn.BatchNorm2d(conv_channel_num, momentum=0.993, eps=1e-5),
            # tf.nn.elu 对应 F.elu 或 nn.ELU
            nn.ELU(),
            # tf.nn.max_pool 对应 nn.MaxPool2d
            nn.MaxPool2d(
                kernel_size=(1, pool_width),
                stride=(1, pool_stride)
            )
        )
        
        # 动态计算CNN输出特征维度以确定LSTM输入大小
        # 创建一个虚拟输入来推断维度
        dummy_input = torch.randn(1, 1, num_node, 400) # window_size=400
        dummy_output = self.conv_block(dummy_input)
        lstm_input_size = int(torch.numel(dummy_output))

        # Bi-LSTM 部分
        # tf.contrib.rnn.LSTMCell 和 bidirectional_dynamic_rnn 对应 nn.LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=n_hidden_state,
            num_layers=2,           # 对应原代码中的 for layer in range(2)
            bidirectional=True,
            batch_first=True,       # PyTorch中推荐使用，使输入维度为 (batch, seq, feature)
            dropout=dropout_prob if 2 > 1 else 0 # 对应 DropoutWrapper
        )
        
        # Attention 部分
        # Bi-LSTM的输出维度是 n_hidden_state * 2
        self.attention = Attention(n_hidden_state * 2)
        
        # Dropout 和 Readout(全连接) 部分
        self.dropout = nn.Dropout(dropout_prob)
        # apply_readout 对应 nn.Linear
        self.fc = nn.Linear(n_hidden_state * 2, num_labels)

    def forward(self, x, batch_size, num_timestep):
        # 输入 x shape: (batch_size * num_timestep, 1, num_node, window_size)
        
        x = self.conv_block(x)
        
        # Flatten and Reshape for LSTM
        x = x.view(x.size(0), -1) # Flatten: (batch_size * num_timestep, features)
        x = x.view(batch_size, num_timestep, -1) # Reshape: (batch_size, num_timestep, features)
        
        # Bi-LSTM
        rnn_output, _ = self.lstm(x) # -> (batch_size, num_timestep, n_hidden_state * 2)
        
        # Attention
        attn_output, _ = self.attention(rnn_output) # -> (batch_size, n_hidden_state * 2)
        
        # Dropout and FC
        x = self.dropout(attn_output)
        logits = self.fc(x) # -> (batch_size, num_labels)
        
        return logits