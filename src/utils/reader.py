import json

import pandas as pd


class Reader:
    base_path: str
    path_trains: dict[str, str] = {}
    path_tests: dict[str, str] = {}
    path_sample_submission: str
    path_submission: str
    rename_map: dict[str, str] = {"participant_index": 'person'}
    rename_set_keys = lambda dct: {Reader.rename_map.get(key, key): value for key, value in dct.items()}

    @classmethod
    def setup(
            cls,
            base_path: str,
            path_x_train: str,
            path_x_test: str,
            path_y_train: str,
            path_y_test: str,
            path_sample_submission: str,
            path_submission: str
    ) -> None:
        cls.base_path = base_path
        cls.path_trains["x"] = base_path + path_x_train
        cls.path_trains["y"] = base_path + path_y_train
        cls.path_tests["x"] = base_path + path_x_test
        cls.path_tests["y"] = base_path + path_y_test
        cls.path_sample_submission = base_path + path_sample_submission
        cls.path_submission = base_path + path_submission

    @staticmethod
    def read_json(path: str) -> dict[str, list[dict[str, str| int]]]:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def get_train_dataset() -> pd.DataFrame:
        y_train_data: pd.DataFrame = Reader.read_csv(Reader.path_trains["y"])

        label_map: dict[str, int] = {
            '01': 0,
            '10': 1,
            '00': 2,
        }

        dataset_train: list[dict[str, list[dict[str, str | int]] | int | str]] = []

        dialog_id: str
        dialog: list[dict[str, str | int]]
        for dialog_id, dialog in Reader.read_json(Reader.path_trains["x"]).items():
            is_bot: list[str] = y_train_data[y_train_data['dialog_id'] == dialog_id][
                ['participant_index', 'is_bot']
            ].sort_values(
                by='participant_index', ascending=True
            )['is_bot'].to_list()

            p0_bot: str = str(is_bot[0])
            p1_bot: str = str(is_bot[1])

            dataset_train.append({
                'dialog': list(map(Reader.rename_set_keys, dialog)),
                'label': label_map[p0_bot + p1_bot],
                'dialog_id': dialog_id,
                'p0_bot': p0_bot,
                'p1_bot': p1_bot,
            })
        return pd.DataFrame(dataset_train)

    @staticmethod
    def get_test_dataset() -> pd.DataFrame:
        return pd.DataFrame([
            {
                'dialog': list(map(Reader.rename_set_keys, dialog)),
                'dialog_id': dialog_id,
            } for dialog_id, dialog in Reader.read_json(Reader.path_tests["x"]).items()
        ])

    @staticmethod
    def get_sample_submission() -> pd.DataFrame:
        return pd.read_csv(Reader.path_sample_submission)

    @staticmethod
    def set_submission(submission: pd.DataFrame) -> None:
        submission.to_csv(Reader.path_submission, index=False)
