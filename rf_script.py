import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

file_path = "scraped_data.json"
with open(file_path, 'r') as f:
    try:
      df = pd.DataFrame(json.load(f))
    except Exception as e:
        print(f"{e}", flush=True)
clf = RandomForestClassifier()
all_characters = set("".join(str(df['plain'].tolist() + df['encoded'].tolist())))
char_encoder = LabelEncoder()
char_encoder.fit(list(all_characters))
df["plain_processed"] = df["plain"].apply(lambda x : char_encoder.transform(list(x)))
df["encoded_processed"] = df["encoded"].apply(lambda x : char_encoder.transform(list(x)))
X_train, X_test, y_train, y_test = train_test_split(df["plain_processed"].tolist(), df["encoded_processed"].to_list(), test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('model.pkl', 'rb') as f:
  clf = pickle.load(f)
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
correct, incorrect = 0, 1
for pred, real in zip(y_pred, y_test):
  same = True
  for a, b in zip(pred, real):
    if a != b:
      same = False
  if same:
    correct += 1
  else:
    incorrect += 1
print(correct / (correct + incorrect))