print("Start import sklearn.metrics")
try:
    from sklearn.metrics import accuracy_score
    print("sklearn.metrics imported successfully")
except Exception as e:
    print(f"Error importing sklearn.metrics: {e}")

print("Start import sklearn.model_selection")
try:
    from sklearn.model_selection import train_test_split
    print("sklearn.model_selection imported successfully")
except Exception as e:
    print(f"Error importing sklearn.model_selection: {e}")
