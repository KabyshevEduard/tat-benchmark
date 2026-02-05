from datasets import Dataset


options = {
    'A': 'anger',
    'B': 'disgust',
    'C': 'fear',
    'D': 'joy',
    'E': 'sadness',
    'F': 'surprise',
    'G': 'none of the above'
}


def correct_answer(emotions_arr):
    if len(emotions_arr) > 0:
        kkey = emotions_arr[0]
        ans_dict = {
            'anger': 'A',
            'disgust': 'B',
            'fear': 'C',
            'joy': 'D',
            'sadness': 'E',
            'surprise': 'F'
        }
        return ans_dict[kkey]
    else:
        return 'G'


def process_docs(dataset: Dataset) -> Dataset:
    def _preprocess_doc(doc):
        question = 'What emotional connotation does this message have?\n'
        dict_item = {
            'question': question + doc['text'],
            'options': options,
            'answer': correct_answer(doc['emotions'])
        }
        return dict_item
    return dataset.map(_preprocess_doc)