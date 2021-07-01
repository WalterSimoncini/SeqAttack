class NERDataset:
    def __init__(self, dataset, label_names: list):
        """
            Initializes a new named entity recognition dataset.

            the dataset is given as a list of tuples in the form
            (text, ner_tags), where ner_tags is an array of class
            labels.
        """
        self.dataset = dataset
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
