import json
from datasets import Dataset
import numpy as np


def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc):
        options = []
        for item in doc['Aspect_VA']:
           options.append(item['Aspect'])

        doc = {
            'text': doc['Text'],
            'options': options,
            'Aspect_VA': doc['Aspect_VA'],
        }
        return doc
    return dataset.map(_process_doc)


def parse_va_string(text: str) -> tuple[float, float]:
    v_a = text.split('#')
    v_a = tuple(map(float, v_a))
    return v_a


def l2(arr_1, arr_2):
    if len(arr_1) != len(arr_2):
        raise ValueError('arr_1 and arr_2 must have same length')
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    decimal = np.sum(np.square((arr_1 - arr_2)))
    return np.sqrt(decimal/len(arr_1))


def rmseva(predictions, references):
    predictions = json.loads(predictions)
    pred_va = []
    ref_va = []
    for pred, truth in zip(predictions, references):
        for aspect_va_p, aspect_va_t in zip(pred, truth):
            valence_t, arousal_t = parse_va_string(aspect_va_t)
            valence_p, arousal_p = parse_va_string(aspect_va_p)
            pred_va.append(valence_p)
            pred_va.append(arousal_p)
            ref_va.append(valence_t)
            ref_va.append(arousal_t)

    return l2(pred_va, ref_va)