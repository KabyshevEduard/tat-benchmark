from datasets import Dataset


def process_docs(dataset: Dataset) -> Dataset:
    ans_dict = {
        'anger': 'A',
        'disgust': 'B',
        'fear': 'C',
        'joy': 'D',
        'sadness': 'E',
        'surprise': 'F'
    }
    correct_answer = lambda x: 'G' if len(x) == 0 else ans_dict[x[0]]
    def _preprocess_doc(doc):
        dict_item = {
            'question': doc['text'],
            'answer': correct_answer(doc['emotions'])
        }
        return dict_item
    return dataset.map(_preprocess_doc)