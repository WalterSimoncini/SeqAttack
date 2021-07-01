# SeqAttack: a framework for adversarial attacks on token classification models

This repository contains the code for "SeqAttack: On Adversarial Attacks For Named Entity Recognition". The framework is contained in the `textattackner` folder, and it is built on top of `TextAttack`. The structure of the framework is very similar to `TextAttack`, and thus the best resource for learning more is the base framework [documentation](https://textattack.readthedocs.io/en/latest/), alongside with the in-code comments in `textattackner`.

### Setup

In order to run the code, I recommend setting up a linux-based virtual machine (only tested on Ubuntu) and running `scripts/gcp.sh`, which sets up an environment with the minimal requirements for running the code. We provide a pre-trained BERT model on CoNLL2003, downloadable at this [link](http://ashita.nl/models/seq-attack/conll2003-ner.tar.gz)

### Tests

Tests can be run with `pytest`

### Runfiles

This repository ships with a few cli files for running experiments and working with datasets, namely:

- `dataset-runner.py`: attacked dataset creation
- `attack-runner.py`: runs attacks on a specified model / dataset / attack strategy combination 
- `cli.py`: model evaluation (CoNLL2003 metrics) and advanced metrics calculation (grammar errors, similarity, ...)
- `metrics-runner.py`: CoNLL2003 metrics calculation for attacked datasets, label utilities for visualization

In addition to these cli scripts, `experiments/analyze-json.py` prints a human-readable representation of an attacked dataset.

### Adversarial examples visualization

The output datasets can be visualized with [SeqAttack-Visualization](https://github.com/WalterSimoncini/SeqAttack-Visualization)