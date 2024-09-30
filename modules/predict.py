"""
Модуль предназначен для загрузки обученной модели
и ее использования для предсказания цены автомобилей
"""

import pandas as pd
import dill
import json
import os
from sklearn.pipeline import Pipeline

path = os.environ.get('PROJECT_PATH', '.')


def load_model(model_dir: str) -> Pipeline:
    """Загружает обученную модель из указанной директории"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    if not model_files:
        raise FileNotFoundError("Файл модели (.pkl) не найден в директории models.")

    model_path = os.path.join(model_dir, model_files[0])
    with open(model_path, 'rb') as file:
        model = dill.load(file)

    return model


def load_test_data(test_data_dir: str) -> pd.DataFrame:
    """Загружает данные для тестирования из указанной директории"""
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.json')]
    test_data = []

    for file in test_files:
        file_path = os.path.join(test_data_dir, file)

        with open(file_path, 'r') as f:
            data = json.load(f)

        df = pd.json_normalize(data)
        df['source_file'] = file
        test_data.append(df)

    return test_data


def make_predictions(model: Pipeline, test_data: pd.DataFrame) -> pd.DataFrame:
    """Делает предсказания на основе модели и тестовых данных"""
    predictions = []

    for df in test_data:
        preds = model.predict(df.drop(columns='source_file'))
        preds_df = pd.DataFrame(preds, columns=['prediction'])
        preds_df['file'] = df['source_file']
        predictions.append(preds_df)

    final_predictions = pd.concat(predictions, ignore_index=True)
    return final_predictions


def save_predictions(predictions: pd.DataFrame, output_dir: str) -> None:
    """Сохраняет предсказания в CSV файл в указанной директории"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions.csv')
    predictions.to_csv(output_path, index=False)


def predict() -> None:
    """Основная функция, которая выполняет полный процесс предсказания"""

    model_dir = os.path.join(path, 'data', 'models')
    model = load_model(model_dir)

    test_data_dir = os.path.join(path, 'data', 'test')
    test_data = load_test_data(test_data_dir)

    predictions = make_predictions(model, test_data)

    output_dir = os.path.join(path, 'data', 'predictions')
    save_predictions(predictions, output_dir)


if __name__ == '__main__':
    predict()
