import json
from datasets import Dataset
import numpy as np
from lm_eval.api.metrics import register_metric
from lm_eval.api.metrics import register_aggregation


def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc):
        options = []
        for item in doc['Aspect_VA']:
           options.append(item['Aspect'])

        doc = {
            'text': doc['Text'],
            'options': options,
            'Aspect_VA': str(doc['Aspect_VA']),
        }
        return doc
    return dataset.map(_process_doc)


def parse_va_string(text: str) -> tuple[float, float]:
    if text is None:
        return 0.0, 0.0
    v_a = text.split('#')
    v_a = tuple(map(float, v_a))
    return v_a


@register_aggregation('rmseva_agg')
def rmseva_agg(value, **kwargs):
    return value


@register_metric(
    metric='rmseva_json',
    higher_is_better=False,
    output_type='generate_until',
    aggregation='rmseva_agg'
)
def rmseva_json(references, predictions):
    all_gold_values = []
    all_pred_values = []
    for gold, pred_str in zip(references, predictions):
        try:
            # Парсим JSON строки
            pred_data = json.loads(pred_str.strip())
            # Извлекаем значения VA
            gold_values = [parse_va_string(item.get('VA')) for item in gold]
            pred_values = [parse_va_string(item.get('VA')) for item in pred_data]
            # Добавление значений в массив
            all_gold_values.extend(gold_values)
            all_pred_values.extend(pred_values)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # В случае ошибки парсинга пропускаем этот пример
            print(f"Ошибка парсинга JSON: {e}")
            continue

    # Проверяем, что есть данные для вычисления
    if len(all_gold_values) != len(all_gold_values):
        return float('inf')
    elif len(all_gold_values) == 0 or len(all_gold_values) == 0:
        return float('inf')

    # Вычисляем
    all_gold_values = np.array(all_gold_values, dtype=float)
    all_pred_values = np.array(all_pred_values, dtype=float)
    result = all_gold_values - all_pred_values
    result = result ** 2
    result = np.sum(result, axis=1)
    mse = np.mean(result)
    rmse = np.sqrt(mse)
    return float(rmse)