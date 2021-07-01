import torch
import numpy as np

from textattack.models.wrappers import ModelWrapper
from textattackner.utils import get_tokens

from textattackner.models.exceptions import PredictionError


class NERModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, postprocess_func):
        self.model = model
        self.tokenizer = tokenizer
        # The post-processing function must accept three arguments:
        #
        # original_text: AttackedText instance of the original text
        # model_output: the model outputs
        # tokenized_input: the tokenized original text
        self.postprocess_func = postprocess_func

    def __call__(self, text_inputs_list, raise_excs=False):
        """
            Get the model predictions for the input texts list

            :param raise_excs:  whether to raise an exception for a
                                prediction error
        """
        encoded = self.encode(text_inputs_list)

        with torch.no_grad():
            outputs = [self._predict_single(x) for x in encoded]

        formatted_outputs = []

        for model_output, original_text in zip(outputs, text_inputs_list):
            tokenized_input = get_tokens(original_text, self.tokenizer)

            # If less predictions than the input tokens are returned skip the sample
            if len(tokenized_input) != len(model_output):
                error_string = f"Tokenized text and model predictions differ in length! (preds: {len(model_output)} vs tokenized: {len(tokenized_input)}) for sample: {original_text}"

                print("Skipping sample")

                if raise_excs:
                    raise PredictionError(error_string)
                else:
                    continue

            formatted_outputs.append(model_output)
    
        return formatted_outputs

    def process_raw_output(self, raw_output, text_input):
        """
            Returns the output as a list of numeric labels
        """
        tokenized_input = get_tokens(text_input, self.tokenizer)

        return self.postprocess_func(
            text_input,
            raw_output,
            tokenized_input)[1]

    def _predict_single(self, encoded_sample):
        outputs = self.model(encoded_sample)[0][0].cpu().numpy()
        return np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

    def get_grad(self, text_input):
        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        self.model.train()
        
        embeddings = self.model.get_input_embeddings()

        original_state = embeddings.weight.requires_grad
        embeddings.weight.requires_grad = True

        embeddings_hook = embeddings.register_backward_hook(grad_hook)
        
        self.model.zero_grad()
        
        encoded_input = self.encode([text_input])[0]
        prediction = self.model(encoded_input).logits
        output = torch.argmax(prediction, dim=2)

        # FIXME: This is how it's done for classification (HuggingFaceModelWrapper).
        # I am really confused on the why you would compare the prediction to the argmax'd
        # prediction itself
        loss = self.model(encoded_input, labels=output)[0]
        loss.backward()
        
        # Gradient of the embeddings
        grad = torch.transpose(emb_grads[0], 0, 1).cpu().numpy()

        # Restore context
        embeddings.weight.requires_grad = original_state
        embeddings_hook.remove()

        self.model.eval()
        
        return {
            "ids": encoded_input[0].tolist(),
            "gradient": grad
        }

    def encode(self, inputs):
        return [self.tokenizer.encode(x, return_tensors="pt") for x in inputs]

    def _tokenize(self, inputs):
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(x, return_tensors="pt"))
            for x in inputs
        ]
