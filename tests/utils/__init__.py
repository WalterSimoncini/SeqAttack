import torch


def numeric_string_to_tensor(in_string):
    """
        Converts a string in the format "0 1 56 7 8"
        to an array tensor
    """
    labels_list = [int(x) for x in in_string.split(" ")]
    return torch.tensor(labels_list)
