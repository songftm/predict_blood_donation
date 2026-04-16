import argparse
import sys
import os

# 将 src 目录添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from train import train
    from predict import main as predict_main
    from evaluate import evaluate_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="献血预测模型运行脚本")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate', 'all'],
                        help='运行模式: train (训练), predict (预测), evaluate (评估), all (全部执行)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=== 开始训练模式 ===")
        train()
    elif args.mode == 'predict':
        print("=== 开始预测模式 ===")
        predict_main()
    elif args.mode == 'evaluate':
        print("=== 开始评估模式 ===")
        evaluate_model()
    elif args.mode == 'all':
        print("=== 执行全流程 ===")
        print("\n--- 步骤 1: 评估模型性能 ---")
        evaluate_model()
        print("\n--- 步骤 2: 全量训练 ---")
        train()
        print("\n--- 步骤 3: 生成预测结果 ---")
        predict_main()

if __name__ == "__main__":
    main()
