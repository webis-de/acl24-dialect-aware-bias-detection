Dialect Aware Social Bias Identification
========================================


Code for the paper [Disentangling Dialect from Social Bias via Multitask Learning to Improve Fairness](https://aclanthology.org/2024.findings-acl.553/).

For details on the approach, architecture and idea, please see the published paper.

```
@inproceedings{spliethover-etal-2024-disentangling,
    title =      "Disentangling Dialect from Social Bias via Multitask Learning to Improve Fairness",
    author =     Splieth{\"o}ver, Maximilian and Menon, Sai Nikhil and Wachsmuth, Henning,
    booktitle =  "Findings of the Association for Computational Linguistics ACL 2024",
    month =      aug,
    year =       "2024",
    address =    "Bangkok, Thailand and virtual meeting",
    publisher =  "Association for Computational Linguistics",
    url =        "https://aclanthology.org/2024.findings-acl.553",
    pages =      "9294--9313",
}
```


## General
- Each directory-related approach contains a `data-preparation.py` script. This should be run before any training scripts.

## Data preparation
- One general and several approach-specific data preparation scripts exist. Run the general preparation script first, then the approach specific scripts.
  - Use the `sbic-data-preparation.ipynb` notebook to prepare TwitterAAE and SBIC corpora.
- The AAE dialect identification is (the last) part of the preprocessing, as later approaches use the annoations of this approach.


## AAE dialect identification
### Data
The dataset published with the paper "Investigating African-American Vernacular English in Transformer-Based Text Generation" is used to train the dialect classifier. The directory `sae-aave-pairs/` from the dataset is expected to be present in `aae-classification/data/`.

### Base models
The basemodel (DeBERTa-v3-large) is expected in `./aae-classification/model`.

### TwitterAAE baseline
The code in the `twitteraae` directory was originally published with Demographic Dialectal Variation in Social Media: A Case Study of African-American English" by Su Lin Blodgett, Lisa Green, and Brendan O'Connor, EMNLP 2016.
We use the code and approach as baseline.

### Script execution order
1. `./aae-classification/data-preparation.py`
2. `./aae-classification/data-splits.py`
3. `./aae-classification/train-weights.sh`
4. `./aae-classification/train-interleaving.sh`


## Finetuning approach
### Data
The approach expects the pre-processed SBIC corpus (see "Social Bias Frames: Reasoning about Social and Power Implications of Language" published by Sap et al. 2022) to present in `./data/sbic-data/`.

### Script execution order
1. `./finetuning/train.sh`
2. `./finetuning/inference-seeded.sh`


## Multitask learning approach
### Data
The approach expects the pre-processed (and with AAE dialect annotated) SBIC corpus (see "Social Bias Frames: Reasoning about Social and Power Implications of Language" published by Sap et al. 2022) to present in `./aae-classification/output/sbic-test_aae-annotated-deberta-v3-base-aee-classifier`.

### Script execution order
1. `./joint-multitask-learning/data-preparation.py`
2. `./joint-multitask-learning/train*.sh` (depending on the approach you want to train)
3. `./joint-multitask-learning/inference*.sh` (depending on the approach trained before)


## Trained models
The trained models evaluated in the paper can be found on huggingface.co :

- [AAE dialect classification models](https://huggingface.co/webis/acl2024-aae-dialect-classification)
- [Social bias classification models](https://huggingface.co/webis/acl2024-social-bias-classification)


## Classification results
The `results/` directory contains the various classification outputs of the paper:

1. The `results/twitteraae-dialect-classification` contains the classification results of the baseline, the weighted loss model, and the data subsampling model on the TwitterAAE dataset.
2. The `results/sbic-bias-classificaiton/` directory contains the bias classification results from all models shown and evaluated in the paper. Each model was run with five different random seeds, as indicated by the `-seedX` postfix of each result file.
3. The SBIC data with AAE dialect annotations, based on our classifier, can be found in the `results/sbic-dialect-classification/` directory.


### Pulling result files
Due to their file size, all result files are stored in the repository using Git LFS, which is a separate package you need to install (detailed instructions avialable [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files)).

To download the result files, run `git lfs checkout` in the repository.
