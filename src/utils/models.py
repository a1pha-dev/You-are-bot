from typing import Any

import catboost as cat
import lightgbm as lgb
import seaborn as sns
import xgboost as xgb

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.text import Text


class Models:
    @staticmethod
    def get_models() -> dict[
        str, lgb.LGBMClassifier | xgb.XGBClassifier | cat.CatBoostClassifier | RandomForestClassifier | Pipeline
    ]:
        return {
            'lgm': lgb.LGBMClassifier(verbose=-1, random_state=42),
            'xgb': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            'cat': cat.CatBoostClassifier(verbose=False, random_state=42),
            'rf': RandomForestClassifier(random_state=42),
            'lr': make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000)
            ),
        }

    @staticmethod
    def get_stats(
            y_true: pd.Series,
            y_pred: pd.Series,
            show_cm: bool = True,
            names: tuple[str, ...] = ('p1 bot', 'p0 bot', 'both human')
    ) -> float | int:
        acc: float | int = accuracy_score(y_true, y_pred)
        report: str | dict = classification_report(y_true, y_pred, target_names=names)
        cm: Any = confusion_matrix(y_true, y_pred)

        print("Accuracy:", acc)
        print("\nClassification Report:\n", report)

        if show_cm:
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()

        return acc

    @staticmethod
    def get_tfidf_probs(
            data_set: pd.DataFrame, vectorizer: TfidfVectorizer, model_tfidf: LogisticRegression
    ) -> tuple[pd.Series, pd.Series]:
        probs0 = model_tfidf.predict_proba(vectorizer.transform(Text.get_texts(data_set, 0)))
        probs1 = model_tfidf.predict_proba(vectorizer.transform(Text.get_texts(data_set, 1)))
        return probs0, probs1

    @staticmethod
    def fit_and_test(
            model: LogisticRegression | lgb.LGBMClassifier | RandomForestClassifier,
            X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
            exp_name: str | None = None
    ) -> tuple[int | float, float]:
        model.fit(X_train, y_train)

        y_pred: pd.Series = model.predict(X_val)
        probs: pd.Series = model.predict_proba(X_val)

        if exp_name is not None:
            print("Exp: ", exp_name)

        logloss: float = log_loss(y_val, probs)
        print(f"Log Loss: {logloss:.4f}")

        acc: int | float = Models.get_stats(y_val, y_pred, show_cm=False, names=('Human', 'Bot'))
        print(acc)

        return acc, logloss
