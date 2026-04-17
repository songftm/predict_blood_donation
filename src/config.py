import torch
import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

    # 数据路径
    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data_blood', 'blood-train.csv')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data_blood', 'blood-test.csv')
    RESULT_PATH = os.path.join(OUTPUT_DIR, 'blood_prediction_results.csv')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'blood_donation_model.pth')

    # 模型参数
    INPUT_SIZE = 7      # 输入特征维度 (4个原始特征 + 3个新特征)
    OUTPUT_SIZE = 1     # 输出维度（二分类）
    HIDDEN_SIZE = 64    # LSTM隐藏层维度
    NUM_LAYERS = 2      # LSTM层数
    NUM_HEADS = 8       # 多头注意力头数 (必须能整除 HIDDEN_SIZE * 2)
    DROPOUT = 0.2       # Dropout比率
    EMBEDDING_DIM = 50  # 嵌入维度（如果使用）

    # 训练参数
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子复现性
    SEED = 42

