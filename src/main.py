from collections.abc import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import lightgbm as lgb
from tqdm import tqdm

from src.utils.environment_settings import EnvironmentTuner
from src.utils.models import Models
from src.utils.reader import Reader
from src.utils.table import Table
from src.utils.text import Text

EnvironmentTuner.set_all_seeds(42)


Reader.setup(
    base_path='../data/',
    path_x_train='train.json',
    path_y_train='ytrain.csv',
    path_x_test='test.json',
    path_y_test='ytest.csv',
    path_sample_submission='sample_submission.csv',
    path_submission='submission.csv'
)
df_train: pd.DataFrame = Reader.get_train_dataset()
df_test: pd.DataFrame = Reader.get_test_dataset()

Text.add_text_features(df_train)
Text.add_text_features(df_test)

y_train: pd.Series = df_train['label'].copy()

X_train: pd.DataFrame = Table.columns_filter(df_train.copy())
X_test: pd.DataFrame = Table.columns_filter(df_test.copy())

X: pd.DataFrame = X_train.copy()
y1, y2 = Table.split_labels(y_train)

X_train2: pd.DataFrame = X.copy()
X_test2: pd.DataFrame = X_test.copy()

names = Models.get_models().keys()
test_probs1: dict[str, np.ndarray] = {name: np.zeros((len(X_test), 2)) for name in names}
test_probs2: dict[str, np.ndarray] = test_probs1.copy()

k: int = len(names)
kfold: KFold = KFold(n_splits=k, shuffle=True, random_state=42)

folds_data: list = []
feature_importance_data: dict[str, list] = {name: [] for name in names}

for fold, (train_index, val_index) in tqdm(enumerate(kfold.split(X)), desc="KFold Progress", total=k):
    X_train_fold = X.iloc[train_index]
    y1_train_fold = y1.iloc[train_index]
    y2_train_fold = y2.iloc[train_index]

    X_val_fold = X.iloc[val_index]
    y1_val_fold = y1.iloc[val_index]
    y2_val_fold = y2.iloc[val_index]

    models1 = Models.get_models()
    models2 = Models.get_models()

    val_acc1: list[int | float] = []
    val_acc2: list[int | float] = []
    val_logloss1: list[float] = []
    val_logloss2: list[float] = []

    name: str
    for name in names:
        tmp_model1 = models1[name]
        tmp_model2 = models2[name]

        tmp_model1.fit(X_train_fold, y1_train_fold)
        tmp_model2.fit(X_train_fold, y2_train_fold)

        probs1: np.ndarray = tmp_model1.predict_proba(X_val_fold)
        probs2: np.ndarray = tmp_model2.predict_proba(X_val_fold)

        val_acc1.append(accuracy_score(y1_val_fold, tmp_model1.predict(X_val_fold)))
        val_acc2.append(accuracy_score(y2_val_fold, tmp_model2.predict(X_val_fold)))
        val_logloss1.append(log_loss(y1_val_fold, probs1))
        val_logloss2.append(log_loss(y2_val_fold, probs2))

        Table.add_probs_to_df(X_train2, probs1, probs2, name, val_index)

        test_probs1[name] += tmp_model1.predict_proba(X_test2) / k
        test_probs2[name] += tmp_model2.predict_proba(X_test2) / k

    folds_data.append({
        'fold': fold + 1,
        'val_acc1': np.mean(val_acc1),
        'val_acc2': np.mean(val_acc2),
        'val_logloss1': np.mean(val_logloss1),
        'val_logloss2': np.mean(val_logloss2),
    })


name: str
for name in names:
    Table.add_probs_to_df(X_test2, test_probs1[name], test_probs2[name], name)

folds_data: pd.DataFrame = pd.DataFrame(folds_data)

X: pd.DataFrame = df_train.copy()

folds_data_tfidf: list[dict[str, int | float]] = []
test_probs1: np.ndarray = np.zeros((len(X_test2), 2))
test_probs2: np.ndarray = test_probs1.copy()

for fold, (train_indexes, val_indexes) in tqdm(enumerate(kfold.split(X_train)), desc="KFold Progress", total=k):
    X_train_fold: pd.DataFrame = X.iloc[train_indexes]
    y1_train_fold: pd.DataFrame = y1.iloc[train_indexes]
    y2_train_fold: pd.DataFrame = y2.iloc[train_indexes]

    X_val_fold: pd.DataFrame = X.iloc[val_indexes]
    y1_val_fold: pd.DataFrame = y1.iloc[val_indexes]
    y2_val_fold: pd.DataFrame = y2.iloc[val_indexes]

    human_texts_train: list[str] = Text.get_human_texts(X_train_fold)
    bot_texts_train: list[str] = Text.get_bot_texts(X_train_fold)

    human_texts_val: list[str] = Text.get_human_texts(X_val_fold)
    bot_texts_val: list[str] = Text.get_bot_texts(X_val_fold)

    y_val_texts: list[int] = [1] * len(bot_texts_val) + [0] * len(human_texts_val)

    vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 3), analyzer='char')
    X_train_vec = vectorizer.fit_transform(bot_texts_train + human_texts_train)
    X_val_vec = vectorizer.transform(bot_texts_val + human_texts_val)

    model_tfidf: LogisticRegression = LogisticRegression(max_iter=1000)
    model_tfidf.fit(X_train_vec, [1] * len(bot_texts_train) + [0] * len(human_texts_train))

    probs1, probs2 = Models.get_tfidf_probs(X_val_fold, vectorizer, model_tfidf)
    Table.add_probs_to_df(X_train2, probs1, probs2, 'tfidf', val_indexes)

    folds_data_tfidf.append({
        'fold': fold + 1,
        'val_acc1': accuracy_score(y1_val_fold, (probs1[:, 1] > 0.5).astype(int)),
        'val_acc2': accuracy_score(y2_val_fold, (probs2[:, 1] > 0.5).astype(int)),
        'val_logloss1': log_loss(y1_val_fold, probs1),
        'val_logloss2': log_loss(y2_val_fold, probs2),
    })

    probs1, probs2 = Models.get_tfidf_probs(df_test, vectorizer, model_tfidf)
    test_probs1 += probs1 / k
    test_probs2 += probs2 / k

Table.add_probs_to_df(X_test2, test_probs1, test_probs2, 'tfidf')

possible_models: dict[
    str, Callable[[], Pipeline] | Callable[[], lgb.LGBMClassifier | Callable[[], RandomForestClassifier]]] = {
    'lr': lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    'lgm': lambda: lgb.LGBMClassifier(verbose=-1, random_state=42),
    'rf': lambda: RandomForestClassifier(random_state=42),
}
chosen_model: str = 'lr'
get_meta_model = possible_models[chosen_model]


X_train2 = Table.get_probs(X_train2)
X_test2 = Table.get_probs(X_test2)

X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
    X_train2, y_train, test_size=0.2, random_state=42
)
label1_train, label2_train = Table.split_labels(y_train_meta)
label1_val, label2_val = Table.split_labels(y_val_meta)


meta_model1 = get_meta_model()
acc1, logloss1 = Models.fit_and_test(
    meta_model1, X_train_meta, label1_train, X_val_meta, label1_val, exp_name='P0 Bot Meta Model'
)

meta_model2 = get_meta_model()
acc2, logloss2 = Models.fit_and_test(
    meta_model2, X_train_meta, label2_train, X_val_meta, label2_val, exp_name='P1 Bot Meta Model'
)

probs1 = meta_model1.predict_proba(X_test2)[:, 1]
probs2 = meta_model2.predict_proba(X_test2)[:, 1]

df = df_test[['dialog_id']].copy()
df['is_bot_0'] = probs1
df['is_bot_1'] = probs2


def get_is_bot(user_id: str):
    global df
    dialog_id, person = user_id.split('_')
    data = df[df['dialog_id'] == dialog_id].reset_index(drop=True)

    if person == '0':
        return data['is_bot_0'][0]
    elif person == '1':
        return data['is_bot_1'][0]
    return None


submission = Reader.get_sample_submission()
submission['is_bot'] = submission['ID'].apply(get_is_bot).astype('float32')
Reader.set_submission(submission)
