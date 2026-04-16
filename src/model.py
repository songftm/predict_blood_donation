import torch
import torch.nn as nn
import torch.nn.functional as F

class BloodDonationLSTM(nn.Module):
    def __init__(self, config):
        """
        初始化LSTM模型
        :param config: 配置对象，包含模型参数
        """
        super(BloodDonationLSTM, self).__init__()
        
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.bidirectional = True # 文档中提到使用双向LSTM
        self.num_directions = 2 if self.bidirectional else 1
        
        # LSTM层
        # input_size: 输入特征维度 (4)
        # hidden_size: 隐藏层维度 (256)
        # num_layers: LSTM层数 (2)
        # batch_first: True, 输入形状为 (batch, seq, feature)
        # bidirectional: True, 双向LSTM
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        
        # 多头注意力机制
        # embed_dim: 注意力机制的输入维度，等于 LSTM 输出维度 (hidden_size * num_directions)
        # num_heads: 注意力头数 (8)
        self.attention_dim = config.HIDDEN_SIZE * self.num_directions
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT
            # 注意：PyTorch旧版本默认 batch_first=False，需要手动调整维度
        )
        
        # 全连接层
        # 输入维度: attention_dim (512)
        # 输出维度: 1 (二分类概率)
        self.fc = nn.Linear(self.attention_dim, config.OUTPUT_SIZE)
        
        # Dropout层
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, input_size)
        :return: 输出概率 (batch_size, 1)
        """
        # LSTM层
        # out: (batch_size, seq_len, hidden_size * num_directions)
        # h_n, c_n: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Attention层
        # MultiheadAttention 默认输入形状为 (seq_len, batch_size, embed_dim)
        # 因此我们需要交换维度 0 和 1
        lstm_out = lstm_out.permute(1, 0, 2) # (seq_len, batch_size, hidden_size*2)
        
        # query, key, value 都是 lstm_out
        # attn_output: (seq_len, batch_size, attention_dim)
        # attn_output_weights: (batch_size, seq_len, seq_len)
        attn_output, attn_output_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 再次交换回 (batch_size, seq_len, attention_dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        # 取最后一个时间步的输出作为上下文向量
        # 或者可以使用所有时间步的加权和，这里根据文档描述 "context_vector = context_vector[:, -1, :]"
        # 由于我们这里 seq_len 可能是 1，取最后一个就是取唯一的那个
        context_vector = attn_output[:, -1, :] 
        
        # 全连接层
        out = self.fc(context_vector)
        
        # Sigmoid激活
        out = self.sigmoid(out)
        
        return out

if __name__ == "__main__":
    # 测试模型结构
    from config import Config
    model = BloodDonationLSTM(Config)
    print(model)
    
    # 测试前向传播
    dummy_input = torch.randn(32, 1, 4) # batch_size=32, seq_len=1, input_size=4
    output = model(dummy_input)
    print("Output shape:", output.shape)
