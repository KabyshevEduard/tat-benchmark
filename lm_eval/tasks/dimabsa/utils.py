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
    value = np.array(value)
    value = np.sum(value)
    value = np.sqrt(value)
    value = float(value)
    return value


@register_metric(
    metric='rmseva_json',
    higher_is_better=False,
    output_type='generate_until',
    aggregation='rmseva_agg'
)
def rmseva_json(references, predictions):
    gold_values = []
    pred_values = []
    for gold, pred_str in zip(references, predictions):
        try:
            # Парсим JSON строки
            pred_data = json.loads(pred_str.strip())
            # Извлекаем значения VA
            gold_values.append(parse_va_string(gold.get('VA')))
            pred_values.append(parse_va_string(pred_data.get('VA')))
            # Добавление значений в массив
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # В случае ошибки парсинга пропускаем этот пример
            print(f"Ошибка парсинга JSON: {e}")
            continue

    # Проверяем, что есть данные для вычисления
    if len(gold_values) != len(pred_values):
        return float('inf')
    elif len(gold_values) == 0 or len(pred_values) == 0:
        r = float('inf')
        return r

    # Вычисляем
    gold_values_np = np.array(gold_values, dtype=float)
    pred_values_np = np.array(pred_values, dtype=float)
    result = gold_values_np - pred_values_np
    result = result ** 2
    result = np.sum(result, axis=1)
    result = np.sum(result)/len(result)
    return float(result)