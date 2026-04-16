print("Start import sklearn.preprocessing")
try:
    from sklearn.preprocessing import StandardScaler
    print("sklearn.preprocessing imported successfully")
except Exception as e:
    print(f"Error importing sklearn.preprocessing: {e}")
