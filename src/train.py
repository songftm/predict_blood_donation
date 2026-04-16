import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_loader import load_and_preprocess_data, get_data_loaders
from model import BloodDonationLSTM
import time

def train():
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    # 1. 加载数据
    print("准备数据...")
    X_train, y_train, X_test, scaler = load_and_preprocess_data()
    
    if X_train is None:
        print("数据加载失败，终止训练。")
        return None

    # 创建数据加载器
    train_loader = get_data_loaders(X_train, y_train, batch_size=Config.BATCH_SIZE)

    # 2. 初始化模型
    print(f"初始化模型... 设备: {Config.DEVICE}")
    model = BloodDonationLSTM(Config).to(Config.DEVICE)
    
    # 3. 定义损失函数和优化器
    criterion = nn.BCELoss() # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. 开始训练
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for i, (batch_features, batch_labels) in enumerate(train_loader):
            # 将数据移动到设备
            batch_features = batch_features.to(Config.DEVICE)
            batch_labels = batch_labels.to(Config.DEVICE)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_features)
            
            # 计算损失
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_features.size(0)
            
        # 计算平均损失
        epoch_loss = epoch_loss / len(train_loader.dataset)
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

    end_time = time.time()
    print(f"训练完成！耗时: {end_time - start_time:.2f} 秒")

    # 5. 保存模型
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"模型已保存至: {Config.MODEL_SAVE_PATH}")
    
    return model, scaler

if __name__ == "__main__":
    train()
