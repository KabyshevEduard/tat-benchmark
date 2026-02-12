from datasets import Dataset


def process_docs(dataset: Dataset) -> Dataset:
    options = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    def _preprocess_doc(doc):
        dict_item = {
            'question': doc['question'],
            'choices': doc['choices'],
            'answer': options[doc['answer']]
        }
        return dict_item
    return dataset.map(_preprocess_doc)