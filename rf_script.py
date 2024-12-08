import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

file_path = "scraped_data.json"
raw_data = ""
with open(file_path, 'r') as f:
    try:
      df = pd.DataFrame(json.loads(f))
    except Exception as e:
        print(f"{e}", flush=True)
clf = RandomForestClassifier()
X = df["plain"]
y = df["encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
with open('rfmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)