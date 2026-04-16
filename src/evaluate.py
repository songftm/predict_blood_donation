import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from config import Config
from model import BloodDonationLSTM
from data_loader import load_and_preprocess_data, get_data_loaders, BloodDonationDataset
from torch.utils.data import DataLoader

# 自定义 train_test_split
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# 自定义 accuracy_score
def custom_accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

# 自定义 confusion_matrix
def custom_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# 自定义 f1_score
def custom_f1_score(y_true, y_pred):
    cm = custom_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model():
    print("开始模型评估流程...", flush=True)
    
    # 1. 加载原始训练数据
    # 注意：这里我们手动进行划分，而不使用 data_loader 中的全量加载
    train_df = pd.read_csv(Config.TRAIN_PATH)
    
    # 提取特征和标签
    X_df = train_df.iloc[:, 1:-1]
    y = train_df.iloc[:, -1].values
    
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
        df['Avg_Frequency'] = months_since_first / (num_donations + 1e-5)

        # 2. 献血密度 = 总献血量 / 首次至今月数
        df['Density'] = total_volume / (months_since_first + 1e-5)

        # 3. 活跃度 = 1 / (上次献血至今月数 + 1)
        df['Activity'] = 1.0 / (months_since_last + 1.0)
        
        return df.values

    X = add_features(X_df)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=Config.SEED)
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 重塑为 LSTM 输入格式 (batch, seq_len=1, input_size)
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_val = X_val.reshape(-1, 1, X_val.shape[1])
    
    # 转为 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    # 创建 DataLoader
    train_loader = get_data_loaders(X_train_tensor, y_train_tensor)
    val_loader = DataLoader(BloodDonationDataset(X_val_tensor, y_val_tensor), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = BloodDonationLSTM(Config).to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 训练模型 (简化版训练循环)
    print("正在验证集上训练模型...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # 评估模型
    print("正在评估模型...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(Config.DEVICE)
            outputs = model(batch_x)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            
    # 计算指标
    acc = custom_accuracy_score(all_labels, all_preds)
    f1 = custom_f1_score(all_labels, all_preds)
    
    # 简单跳过AUC计算，或者如果你需要，我可以尝试实现一个简易版
    auc = "N/A (sklearn unavailable)"
        
    cm = custom_confusion_matrix(all_labels, all_preds)
    
    print("-" * 30)
    print("评估结果 (验证集):")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")
    print(f"AUC-ROC: {auc}")
    print("混淆矩阵:")
    print(cm)
    print("-" * 30)

if __name__ == "__main__":
    evaluate_model()
