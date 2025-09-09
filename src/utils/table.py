import numpy as np
import pandas as pd


class Table:
    @staticmethod
    def columns_filter(dataset: pd.DataFrame) -> pd.DataFrame:
        blacklist: list[str] = [
            'dialog', 'p0_messages', 'p1_messages',
            'label', 'p0_bot', 'p1_bot', 'dialog_id',
        ]
        return dataset[filter(lambda col: col not in blacklist, dataset.columns.tolist())]

    @staticmethod
    def split_labels(series: pd.Series) -> tuple[pd.Series, pd.Series]:
        return (series == 1).astype(int), (series == 0).astype(int)

    @staticmethod
    def add_probs_to_df(
            dataset: pd.DataFrame,
            probabilities_1: np.ndarray,
            probabilities_2: np.ndarray,
            name: str, indexes: list[int] | None = None
    ) -> None:
        if indexes is None:
            indexes = dataset.index.tolist()

        i: int
        for i in 0, 1:
            if f'p{i}_prob_{name}' not in dataset.columns:
                dataset[f'p{i}_prob_{name}'] = np.nan
        dataset.loc[dataset.index[indexes], f'p0_prob_{name}'] = probabilities_1[:, 1]
        dataset.loc[dataset.index[indexes], f'p1_prob_{name}'] = probabilities_2[:, 1]

    @staticmethod
    def get_probs(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[
            filter(lambda col: col.startswith('p0_prob_') or col.startswith('p1_prob_'), dataset.columns.tolist())
        ]
