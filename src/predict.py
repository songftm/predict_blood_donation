import torch
import pandas as pd
import numpy as np
from config import Config
from model import BloodDonationLSTM
from data_loader import load_and_preprocess_data

def predict(model, X_test):
    """
    使用模型进行预测
    :param model: 训练好的模型
    :param X_test: 测试数据张量
    :return: 预测概率
    """
    model.eval()
    with torch.no_grad():
        # 确保数据在正确的设备上
        X_test = X_test.to(Config.DEVICE)
        
        # 前向传播
        outputs = model(X_test)
        
        # 获取预测概率
        predictions = outputs.cpu().numpy()
        
    return predictions

def main():
    print("开始预测流程...")
    
    # 1. 加载数据
    # 注意：我们需要加载训练数据来拟合Scaler，以确保对测试集的标准化是一致的
    # 虽然这里有点低效（重新读取和拟合），但保证了正确性
    # 在生产环境中，应该保存并加载Scaler对象
    _, _, X_test, _ = load_and_preprocess_data()
    
    if X_test is None:
        print("数据加载失败。")
        return

    # 2. 加载模型
    print(f"加载模型: {Config.MODEL_SAVE_PATH}")
    
    # 动态获取模型输入维度，确保与训练时一致
    try:
        # 尝试从数据维度推断
        input_size = X_test.shape[2] 
        print(f"推断出的输入特征维度: {input_size}")
        # 临时覆盖Config中的INPUT_SIZE，以防config未更新
        Config.INPUT_SIZE = input_size
    except Exception as e:
        print(f"无法推断输入维度，使用Config默认值: {Config.INPUT_SIZE}")

    model = BloodDonationLSTM(Config)
    try:
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
        model.to(Config.DEVICE)
        model.eval() # 切换到评估模式
    except FileNotFoundError:
        print("错误：找不到模型文件。请先运行 train.py 进行训练。")
        return
    except RuntimeError as e:
        print(f"模型加载错误: {e}")
        print("可能原因：模型结构（如隐藏层大小）与保存时不一致，或者输入维度不匹配。")
        return

    # 3. 进行预测
    print("正在进行预测...")
    try:
        predictions = predict(model, X_test)
    except RuntimeError as e:
        print(f"预测运行时错误: {e}")
        return
    
    # 4. 保存结果
    try:
        # 读取原始测试文件以保留ID等信息
        test_df = pd.read_csv(Config.TEST_PATH)
        
        # 将预测结果添加到DataFrame
        # 假设最后一列是预测目标，或者我们需要新建一列
        # 文档中提到 'Made Donation in March 2007' 是目标列
        # predictions 是 (N, 1) 形状的数组，需要展平
        test_df['Made Donation in March 2007'] = predictions.flatten()
        
        # 保存
        test_df.to_csv(Config.RESULT_PATH, index=False)
        print(f"预测结果已保存至: {Config.RESULT_PATH}")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    main()
