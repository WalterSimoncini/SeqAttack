from transformers import AutoTokenizer, AutoModelForTokenClassification

from textattackner.models import NERModelWrapper
from textattackner.utils import postprocess_ner_output


def get_ner_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return tokenizer, NERModelWrapper(
        model,
        tokenizer,
        postprocess_func=postprocess_ner_output)
