import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

file_path = "scraped_data.json"
raw_data = ""
with open(file_path, 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            raw_data += line
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {i}: {e}")
df = pd.DataFrame(json.loads(raw_data))
clf = RandomForestClassifier()
X = df["plain"]
y = df["encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
with open('rfmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)