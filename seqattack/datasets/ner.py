import copy
import random


class NERDataset:
    def __init__(
        self,
        dataset,
        label_names: list,
        num_examples: int = None,
        dataset_name: str = None):
        """
            Initializes a new named entity recognition dataset.

            the dataset is given as a list of tuples in the form
            (text, ner_tags), where ner_tags is an array of class
            labels (e.g. [0, 0, 2, ...]).
        """
        if num_examples:
            self.dataset = dataset[:num_examples]
        else:
            self.dataset = dataset

        self.dataset_name = dataset_name
        self.current_idx = 0
        self.label_names = label_names

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < len(self.dataset):
            self.current_idx += 1

            return self.dataset[self.current_idx - 1]
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    def to_dict(self):
        predictions_mappings = {}

        for sentence, prediction in self.dataset:
            predictions_mappings[sentence] = prediction

        return predictions_mappings

    def filter(self, filter_function):
        self.dataset = list(filter(filter_function, self.dataset))

    def shuffle(self):
        random.shuffle(self.dataset)

    def tolist(self):
        return copy.deepcopy(self.dataset)

    @property
    def name(self):
        """The dataset name if any"""
        return self.dataset_name

    @property
    def split(self):
        """The dataset split if any"""
        return None
