# SeqAttack: a framework for adversarial attacks on token classification models

SeqAttack is a framework for conducting adversarial attacks against Named Entity Recognition (NER) models and for data augmentation. This library is heavily based on the popular [TextAttack](https://github.com/QData/TextAttack) framework, and can similarly be used for:

- Understanding models by running adversarial attacks against them and observing their shortcomings
- Develop new attack strategies
- Guided data augmentation, generating additional training samples that can be used to fix a model's shortcomings

### Setup

Run `pip install -r requirements.txt` and you're good to go! If you want to run experiments on a fresh virtual machine, check out `scripts/gcp.sh` which installs all system dependencies for running the code. 

The code was tested with `python 3.7`, if you're using a different version your mileage may vary.

### Usage

The main features of the framework are available via the command line interface, wrapped by `cli.py`. The following subsections describe the usage of the various commands.

#### Attack

Attacks are executed via the `python cli.py attack` subcommand. Attack commands are split in two parts:

- General setup: options common to all adversarial attacks (e.g. model, dataset...)
- Attack specific setup: options specific to a particular attack strategy

Thus, a typical attack command might look like the following:

```sh
python cli.py attack [general-options] attack-recipe [recipe-options]
```

For example, if we want to attack [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER), a NER model trained on CoNLL2003 using `deepwordbug` as the attack strategy we might run:

```sh
python cli.py attack                                            \
       --model-name dslim/bert-base-NER                         \
       --output-path output-dataset.json                        \
       --cache                                                  \
       --dataset-config configs/conll2003-config.json           \
       deepwordbug
```

The dataset configuration file, `configs/conll2003-config.json` defines:

- The dataset path or name (in the latter case it will be downloaded from [HuggingFace](https://huggingface.co/datasets))
- The split (e.g. train, test). Only for HuggingFace datasets
- The human-readable names (a mapping between numerical labels and textual labels), given as a list
- A `labels map`, used to remap the dataset's ground truth to align it with the model output as needed. This field can be `null` if no remapping is needed

In the example above, `labels_map` is used to align the dataset labels to the output from `dslim/bert-base-NER`. The dataset labels are the following:

`O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4) B-LOC (5), I-LOC (6) B-MISC (7), I-MISC (8)`

while the model labels are:

`O (0), B-MISC (1), I-MISC (2), B-PER (3), I-PER (4) B-ORG (5), I-ORG (6) B-LOC (7), I-LOC (8)`

Thus a remapping is needed and `labels_map` takes care of it.

---

The available attack strategies are the following:

| Attack Strategy | Transformation                                                   | Constraints                                                        | Paper                                                  |
|-----------------|------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------|
| BAE             | word swap                                                        | USE sentence cosine similarity                                     | https://arxiv.org/abs/2004.01970                       |
| BERT-Attack     | word swap                                                        | USE sentence cosine similarity, Maximum words perturbed            | https://arxiv.org/abs/2004.09984                       |
| CLARE           | word swap and insertion                                          | USE sentence cosine similarity                                     | https://arxiv.org/abs/2009.07502                       |
| DeepWordBug     | character insertion, deletion, swap (ab --> ba) and substitution | Levenshtein edit distance                                          | https://arxiv.org/abs/1801.04354                       |
| Morpheus        | inflection word swap                                             |                                                                    | https://www.aclweb.org/anthology/2020.acl-main.263.pdf |
| SCPN            | paraphrasing                                                     |                                                                    | https://www.aclweb.org/anthology/N18-1170              |
| TextFooler      | word swap                                                        | USE sentence cosine similarity, POS match, word-embedding distance | https://arxiv.org/abs/1907.11932                       |

The table above is based on [this table](https://github.com/QData/TextAttack#attacks-and-papers-implemented-attack-recipes-textattack-attack---recipe-recipe_name). In addition to the constraints shown above the attack strategies **are also forbidden from modifying and inserting named entities by default**.

#### Evaluation

To evaluate a model against a standard dataset run:

```sh
python cli.py evaluate                  \
       --model dslim/bert-base-NER      \
       --dataset conll2003              \
       --split test                     \
       --mode strict                    \
```

To evaluate the effectivenes of an attack run the following command:

```sh
python cli.py evaluate                                  \
       --model dslim/bert-base-NER                      \
       --attacked-dataset experiments/deepwordbug.json  \
       --mode strict                                    \
```

The above command will compute and display the metrics for the original predictions and their adversarial counterparts.

The evaluation is based on [seqeval](https://github.com/chakki-works/seqeval)

#### Dataset selection

Given a dataset, our victim model may be able to predict some dataset samples perfectly, but it may produce significant errors on others. To evaluate an attack's effectiveness we may want to select samples with a small initial misprediction score. This can be done via the following command:

```sh
python cli.py pick-samples                              \
       --model dslim/bert-base-NER                      \
       --dataset-config configs/conll2003-config.json   \
        --max-samples 256                               \
       --max-initial-score 0.5                          \ # The maximum initial misprediction score
       --output-filename cherry-picked.json             \
       --goal-function untargeted
```


### Tests

Tests can be run with `pytest`

### Adversarial examples visualization

The output datasets can be visualized with [SeqAttack-Visualization](https://github.com/WalterSimoncini/SeqAttack-Visualization)