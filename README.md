# 基于LSTM的献血预测模型 (Blood Donation Prediction using LSTM)

这是一个基于PyTorch实现的双向LSTM模型，用于预测献血者是否会在特定时间点（2007年3月）进行献血。该项目参考了彩票预测系统的实现方式，并针对献血数据进行了适配。

## 1. 项目结构

```
predict_blood_donation/
├── src/                # [源代码] 核心业务逻辑
│   ├── config.py       # [核心配置] 所有超参数和路径配置
│   ├── data_loader.py  # [数据处理] 数据加载、清洗、标准化
│   ├── model.py        # [模型定义] Bi-LSTM + Attention 模型
│   ├── train.py        # [训练脚本] 模型训练循环
│   ├── evaluate.py     # [评估脚本] 模型性能评估
│   └── predict.py      # [预测脚本] 结果预测
├── output/             # [输出文件] 模型权重和预测结果
│   ├── blood_donation_model.pth
│   └── blood_prediction_results.csv
├── tests/              # [测试脚本] 单元测试和环境检查
├── main.py             # [主入口] 一键运行各模块
└── requirements.txt    # 项目依赖
```

## 2. 环境准备

确保你已经安装了 Python 3.7+。

### 安装依赖库
在项目根目录下运行以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```
如果需要安装 PyTorch（通常 pip install torch 即可，如果需要 GPU 版本请参考 PyTorch 官网）：
```bash
pip install torch
```

## 3. 快速开始

### 3.1 检查配置
打开 `config.py` 文件，确认 `TRAIN_PATH` 和 `TEST_PATH` 指向正确的数据文件路径。
当前默认路径为：
- 训练集: `d:\fushi_work\predict_Lottery_ticket_pytorch-master\predict_Lottery_ticket_pytorch-master\data_blood\blood-train.csv`
- 测试集: `d:\fushi_work\predict_Lottery_ticket_pytorch-master\predict_Lottery_ticket_pytorch-master\data_blood\blood-test.csv`

### 3.2 运行模型
你可以通过 `main.py` 脚本以不同的模式运行项目。

**模式说明：**
- `train`: 训练模型并保存到 `blood_donation_model.pth`
- `evaluate`: 在训练集上划分验证集，评估模型性能 (Accuracy, F1, AUC)
- `predict`: 加载训练好的模型，对测试集进行预测，结果保存为 `blood_prediction_results.csv`
- `all`: 依次执行 评估 -> 训练 -> 预测 全流程

#### 示例命令：

1. **评估模型性能** (查看准确率等指标):
   ```bash
   python main.py --mode evaluate
   ```

2. **训练模型** (使用全部数据训练):
   ```bash
   python main.py --mode train
   ```

3. **生成预测结果**:
   ```bash
   python main.py --mode predict
   ```

4. **一键全流程**:
   ```bash
   python main.py --mode all
   ```

## 4. 模型说明
- **输入特征**: 
  1. Months since Last Donation (上次献血至今月数)
  2. Number of Donations (献血次数)
  3. Total Volume Donated (总献血量)
  4. Months since First Donation (首次献血至今月数)
- **模型架构**:
  - 输入层 -> 双向 LSTM (2层, hidden_size=256) -> Dropout -> Multihead Attention (8 heads) -> 全连接层 -> Sigmoid 输出
- **输出**: 预测概率 (0~1)，表示献血的可能性。

## 5. 结果文件
- 模型文件: `blood_donation_model.pth`
- 预测结果: `blood_prediction_results.csv`
