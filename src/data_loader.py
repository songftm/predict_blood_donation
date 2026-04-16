import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import Config
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

class BloodDonationDataset(Dataset):
    """
    自定义数据集类，用于加载和处理献血数据。
    继承自 torch.utils.data.Dataset。
    """
    def __init__(self, features, labels=None):
        """
        初始化数据集。
        :param features: 输入特征张量
        :param labels: 标签张量（可选，测试集可能没有标签）
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        feature = self.features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return feature, label
        return feature

def load_and_preprocess_data():
    """
    加载并预处理数据。
    流程包括：
    1. 读取CSV文件
    2. 提取特征和标签
    3. 数据标准化（StandardScaler）
    4. 转换为LSTM所需的序列格式 (batch_size, seq_len=1, input_size)
    5. 转换为PyTorch张量
    """
    print("正在加载数据...")
    # 1. 读取CSV文件
    try:
        train_df = pd.read_csv(Config.TRAIN_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件。请检查路径配置。{e}")
        return None, None, None, None

    # 2. 提取特征和标签
    # 训练集：第一列是ID（通常不需要），最后一列是标签
    # 假设第一列是索引列，或者是无关列，我们需要根据实际数据列名调整
    # 根据文档：Months since Last Donation, Number of Donations, Total Volume Donated (c.c.), Months since First Donation
    # 以及 Made Donation in March 2007
    
    # 检查列名以确保正确提取
    print("训练集列名:", train_df.columns.tolist())
    
    # 通常第一列可能是ID，最后一列是标签
    # 如果第一列是ID，我们应该跳过它。如果第一列就是特征，则不跳过。
    # 这里我们假设第一列是ID（常见情况），如果不是，可以调整 iloc[:, 1:-1]
    # 文档中的代码示例使用了 iloc[:, 1:-1]，我们遵循这个假设，但添加打印以确认
    
    # 特征提取 (根据文档示例：跳过第一列ID，去掉最后一列标签)
    X_train_df = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1].values
    
    # 测试集特征 (根据文档示例：跳过第一列ID)
    X_test_df = test_df.iloc[:, 1:]

    # --- 特征工程 ---
    print("正在进行特征工程...", flush=True)
    def add_features(df):
        # 复制一份以免修改原数据
        df = df.copy()
        
        # 确保列是数值类型
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 使用 iloc 访问列，避免列名问题
        months_since_last = df.iloc[:, 0]
        num_donations = df.iloc[:, 1]
        total_volume = df.iloc[:, 2]
        months_since_first = df.iloc[:, 3]

        # 1. 平均献血频率 = 首次至今月数 / 献血次数
        # 频率越高（数值越小），表示献血越频繁
        df['Avg_Frequency'] = months_since_first / (num_donations + 1e-5)

        # 2. 献血密度 = 总献血量 / 首次至今月数
        # 密度越大，表示单位时间内贡献越大
        df['Density'] = total_volume / (months_since_first + 1e-5)

        # 3. 活跃度 = 1 / (上次献血至今月数 + 1)
        # 数值越大（越接近1），表示最近刚献过血，越活跃
        df['Activity'] = 1.0 / (months_since_last + 1.0)
        
        return df.values

    X_train = add_features(X_train_df)
    X_test = add_features(X_test_df)
    print(f"特征工程完成。新特征维度: {X_train.shape[1]}", flush=True)

    # 3. 数据标准化
    # 使用训练集的统计量来标准化测试集，防止数据泄露
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3.5 SMOTE过采样 (如果安装了imblearn)
    # 注意：SMOTE 应该在标准化之后、序列重塑之前进行
    # 也可以在标准化之前，但通常建议对数值特征标准化后再处理
    if SMOTE is not None:
        # try:
        #     print("正在应用 SMOTE 进行过采样...", flush=True)
        #     # 统计标签分布，注意 y_train 可能是浮点数，需要转为整数统计
        #     unique, counts = np.unique(y_train.astype(int), return_counts=True)
        #     print(f"原始训练集形状: {X_train_scaled.shape}, 标签分布: {dict(zip(unique, counts))}", flush=True)
            
        #     smote = SMOTE(random_state=Config.SEED)
        #     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
        #     unique_res, counts_res = np.unique(y_train_resampled.astype(int), return_counts=True)
        #     print(f"过采样后训练集形状: {X_train_resampled.shape}, 标签分布: {dict(zip(unique_res, counts_res))}", flush=True)
            
        #     # 使用过采样后的数据
        #     X_train_scaled = X_train_resampled
        #     y_train = y_train_resampled
        # except Exception as e:
        #     print(f"SMOTE 过采样失败: {e}", flush=True)
        print("已暂时禁用 SMOTE，优先测试特征工程效果。", flush=True)
    else:
        print("未检测到 imbalanced-learn 库，跳过 SMOTE 过采样。", flush=True)

    # 4. 构建序列数据
    # LSTM 需要输入形状为 (batch_size, sequence_length, input_size)
    # 对于这种表格数据，我们将其视为序列长度为 1 的时间序列
    X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

    # 5. 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1) # 变为列向量
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

    print(f"训练集形状: {X_train_tensor.shape}, 标签形状: {y_train_tensor.shape}")
    print(f"测试集形状: {X_test_tensor.shape}")

    return X_train_tensor, y_train_tensor, X_test_tensor, scaler

def get_data_loaders(X_train, y_train, batch_size=Config.BATCH_SIZE):
    """
    创建数据加载器
    """
    dataset = BloodDonationDataset(X_train, y_train)
    # 在实际项目中，通常会划分验证集。这里为了简单演示，直接使用全部训练集
    # 更好的做法是使用 train_test_split 划分
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == "__main__":
    # 测试代码
    X_train, y_train, X_test, scaler = load_and_preprocess_data()
    if X_train is not None:
        loader = get_data_loaders(X_train, y_train)
        for batch_x, batch_y in loader:
            print("Batch X shape:", batch_x.shape)
            print("Batch Y shape:", batch_y.shape)
            break
