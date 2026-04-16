import pandas as pd
try:
    path = r'd:\fushi_work\predict_Lottery_ticket_pytorch-master\predict_Lottery_ticket_pytorch-master\data_blood\blood-train.csv'
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("Head:", df.head())
except Exception as e:
    print(f"Error: {e}")
