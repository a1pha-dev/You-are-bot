import json

import nltk

import pandas as pd
from sklearn import metrics

nltk.download("stopwords")
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


russian_stopwords: list[str] = stopwords.words("russian")


def read_data(path: str) -> dict[str, str]:
    raw_data: dict[str, list[dict[str, str | int]]] = json.load(open(path))
    data: dict[str, str] = {}

    key: str
    value: str
    for key, value in raw_data.items():
        data[f"{key}_0"] = " ".join(list(map(lambda message: message["text"], value[::2])))
        data[f"{key}_1"] = "".join(list(map(lambda message: message["text"], value[1::2])))

    return data


data = read_data("../data/train.json")

answers = pd.read_csv("../data/ytrain.csv")
answers["user_id"] = answers["dialog_id"].map(str).add("_").add(answers["participant_index"].map(str))
answers["text"] = answers["user_id"].map(data)

X_train, X_valid, y_train, y_valid = train_test_split(answers["text"], answers['is_bot'], test_size=0.1,
                                                      random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

knb_ppl_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000)),
    ('knb_clf', RandomForestClassifier(max_depth=30, max_features="log2", n_estimators=10000, class_weight="balanced",
                                       random_state=42))])

knb_ppl_clf.fit(X_train, y_train)
predicted_sgd = knb_ppl_clf.predict(X_test)
print(metrics.classification_report(predicted_sgd, y_test))

data = read_data("../data/test.json")

predicted_sgd = knb_ppl_clf.predict_proba(data.values())

result = pd.DataFrame({"ID": data.keys(), "is_bot": map(lambda note: note[1], predicted_sgd)})
result.to_csv("../data/submission.csv", index=False)
