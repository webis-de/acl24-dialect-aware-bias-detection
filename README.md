Dialect Aware Social Bias Identification
========================================


Code for the paper [Disentangling Dialect from Social Bias via Multitask Learning to Improve Fairness](https://arxiv.org/abs/2406.09977).

For details on the approach, architecture and idea, please see the published paper.

NOTE: ALL PRE_TRAINED MODELS AND CLASSIFICATION OUTPUTS WILL BE ADDED SOON. :)

```
@InProceedings{spliethoever2024,
  address =                  {Bangkok, Thailand},
  author =                   {Maximilian Splieth{\"o}ver and Sai Nikhil Menon and Henning Wachsmuth},
  booktitle =                {Findings of the Association for Computational Linguistics: ACL 2024},
  month =                    aug,
  publisher =                {Association for Computational Linguistics},
  title =                    {{Disentangling Dialect from Social Bias via Multitask Learning to Improve Fairness}},
  url =                      {https://arxiv.org/abs/2406.09977},
  year =                     2024
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
