import copy

from OpenAttack.attackers import SCPNAttacker

from textattack.transformations import Transformation
from textattackner.utils.ner_attacked_text import NERAttackedText


class ParaphraseTransformation(Transformation):
    """
        Generates paraphrases of the input sentence,
        re-mapping the original ground truth and making
        sure named entities are preserved.

        Based on (as implemented in OpenAttack):

        Adversarial Example Generation with Syntactically Controlled
        Paraphrase Networks.

        Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer.
        NAACL-HLT 2018.

        `[pdf] <https://www.aclweb.org/anthology/N18-1170.pdf>`__
        `[code] <https://github.com/miyyer/scpn>`__
    """
    def __init__(self):
        self.attacker = SCPNAttacker()

    def _get_transformations(self, current_text, indices_to_modify):
        """
            Returns paraphrases of the input text, mapping named
            entities to the new ground truth.

            WARNING: indices_to_modify is ignored in this transformation
        """
        # Extract entities from the input text

        # FIXME: this strategy might have problems
        # if we have two named entities with the same
        # name and a different label
        entities = {}
        tokens = current_text.text.split(" ")
        ground_truth = current_text.attack_attrs["ground_truth"]

        for token, truth in zip(tokens, ground_truth):
            if truth == 0:
                continue

            entities[token.lower()] = {
                "token": token,
                "truth": truth
            }

        entities_set = set(entities.keys())

        candidates = self.attacker.gen_paraphrase(
            current_text.text,
            self.attacker.config["templates"]
        )

        out_texts = []

        for cnd in candidates:
            cnd_tokens = cnd.split(" ")

            if not entities_set.issubset(set(cnd_tokens)):
                # All entity token must still be there
                continue

            # Sample approved, remap the ground truth
            final_cnd_tokens, cnd_truth = [], []

            for cnd_token in cnd_tokens:
                if cnd_token in entities:
                    # Label named entities in the transformed text and
                    # preserve capitalization
                    final_cnd_tokens.append(entities[cnd_token]["token"])
                    cnd_truth.append(entities[cnd_token]["truth"])
                else:
                    # All other tokens are considered as having no class
                    final_cnd_tokens.append(cnd_token)
                    cnd_truth.append(0)

            attack_attrs = copy.deepcopy(current_text.attack_attrs)
            attack_attrs["ground_truth"] = cnd_truth

            final_text = " ".join(final_cnd_tokens)

            out_texts.append(
                NERAttackedText(
                    final_text,
                    attack_attrs=attack_attrs
                )
            )

        return out_texts
