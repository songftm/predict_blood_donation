print("Start import sklearn")
try:
    from sklearn.model_selection import train_test_split
    print("sklearn imported successfully")
except Exception as e:
    print(f"Error importing sklearn: {e}")
