from datasets import Dataset


def process_docs(dataset: Dataset) -> Dataset:
    return dataset.shuffle(seed=42).select(range(10000))