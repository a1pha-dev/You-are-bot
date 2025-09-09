import pandas as pd
import numpy as np

import emoji

import re


class Text:
    @staticmethod
    def add_dialog_length(dataset: pd.DataFrame) -> None:
        dataset['dialog_length'] = dataset['dialog'].apply(len)

    @staticmethod
    def add_messages_by_persons(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_messages'] = dataset['dialog'].apply(
                lambda x: [msg['text'] for msg in x if msg['person'] == str(i)]
            )

    @staticmethod
    def count_emoji(text: str) -> int:
        i: int
        return sum(text.count(symbol) for symbol in emoji.EMOJI_DATA)

    @staticmethod
    def add_emoji_count(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_emoji_count'] = dataset[f'p{i}_text'].apply(Text.count_emoji)

    @staticmethod
    def add_emoji_percent(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_messages'].apply(
                lambda x: sum(Text.count_emoji(msg) > 0 for msg in x) / len(x) if len(x) > 0 else 0
            )

    @staticmethod
    def add_text(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_text'] = dataset[f'p{i}_messages'].apply(lambda sentence: ' '.join(sentence))

    @staticmethod
    def add_mean_length(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_mean_length'] = dataset[f'p{i}_messages'].apply(lambda x: np.mean(list(map(len, x))))

    @staticmethod
    def add_special_count(dataset: pd.DataFrame) -> None:
        special_chars: list[str] = ['?', '!']

        i: int
        for i in 0, 1:
            dataset[f'p{i}_special_count'] = dataset[f'p{i}_text'].apply(
                lambda x: sum(x.count(char) for char in special_chars) / len(x) if len(x) > 0 else 0
            )

    @staticmethod
    def add_first_capital_percent(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_first_capital_percent'] = dataset[f'p{i}_messages'].apply(
                lambda x: sum(len(msg) > 0 and msg[0].isupper() for msg in x) / len(x) if len(x) > 0 else 0
            )

    @staticmethod
    def add_capital_percent(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_capital_percent'] = dataset[f'p{i}_text'].apply(
                lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0
            )

    @staticmethod
    def add_mean_words_length(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_mean_words_length'] = dataset[f'p{i}_text'].apply(
                lambda x: np.mean(list(map(len, x.split()))) if len(x) > 0 else 0
            )

    @staticmethod
    def add_mean_words_count(dataset: pd.DataFrame) -> None:
        i: int
        for i in 0, 1:
            dataset[f'p{i}_mean_words_count'] = dataset[f'p{i}_text'].apply(
                lambda x: np.mean(list(map(lambda msg: len(msg.split()), x.split('\n')))) if len(x) > 0 else 0
            )

    @staticmethod
    def echo_count(dialog: list[dict[str, str | int]]) -> tuple[int, int]:
        echo_counts: dict[str, int] = {"0": 0, "1": 0}
        prev_text: str | None = None

        msg: dict[str, str | int]
        for msg in dialog:
            text: str = msg['text']
            person: str = msg['person']
            if prev_text and text == prev_text:
                echo_counts[person] += 1
            prev_text = text
        return echo_counts['0'], echo_counts['1']

    @staticmethod
    def add_echo_bot_index(data_set: pd.DataFrame) -> None:
        echo_counts: dict[int, pd.Series] = {
            i: data_set['dialog'].apply(lambda x: Text.echo_count(x)[i]) for i in (0, 1)
        }

        i: int
        for i in 0, 1:
            data_set[f'p{i}_echo_index'] = echo_counts[i] / data_set[f'p{i}_messages'].apply(
                lambda x: len(x) - 1 + (len(x) > 1)
            )

    @staticmethod
    def add_text_features(dataset: pd.DataFrame) -> None:
        for func in [
            Text.add_dialog_length, Text.add_messages_by_persons, Text.add_text, Text.add_emoji_count,
            Text.add_emoji_percent, Text.add_mean_length, Text.add_mean_words_length, Text.add_mean_words_count,
            Text.add_special_count, Text.add_first_capital_percent, Text.add_capital_percent, Text.add_echo_bot_index
        ]:
            func(dataset)
        dataset.drop(columns=['p0_text', 'p1_text'], inplace=True)

    @staticmethod
    def get_texts(dataset: pd.DataFrame, person: int) -> list[str]:
        return dataset[f'p{person}_messages'].apply(lambda x: Text.preprocess('\n'.join(x))).to_list()

    @staticmethod
    def get_bot_texts(dataset: pd.DataFrame) -> list[str]:
        texts0: list[str] = Text.get_texts(dataset[dataset['p0_bot'] == '1'], 0)
        texts1: list[str] = Text.get_texts(dataset[dataset['p1_bot'] == '1'], 1)
        return texts0 + texts1

    @staticmethod
    def get_human_texts(dataset: pd.DataFrame) -> list[str]:
        texts0: list[str] = Text.get_texts(dataset[dataset['p0_bot'] == '0'], 0)
        texts1: list[str] = Text.get_texts(dataset[dataset['p1_bot'] == '0'], 1)
        return texts0 + texts1

    @staticmethod
    def preprocess(text):
        return re.sub(r'[^\w\s]', '',
                      re.sub(r'\s+', ' ',
                             re.sub(r'https?://\S+|www\.\S+', '', text.lower()))).strip()
