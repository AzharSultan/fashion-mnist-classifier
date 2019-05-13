# fashion-mnist-classifier

This repository contains a minivgg classifier trained on fashion-mnist dataset that achieves an accuracy of 95.38%

## Getting Started
The code has been tested with **cond 4.3.30** and **python 3.6.8**. Run the following commands to setup the environment:

```
conda create -n fashion-mnist
pip install -r requirements.txt
```
### Training
Training can be run using the following command:
`python train.py --config_file config/train.yml`

Default parameters in the train.yml are the ones that achieved the best result. Parameters that are not self-explanatory are:
- **random_labels**: Set True to train with random training labels
- **hyperparameter_optimization**: done using hyperas, with values to optimize between hard-coded in train.py
- **first_block**: the number of kernels used in the first convolution block of minivgg

### Evaluation
Evaluation on the best model can be run using:
`python evaluate.py --config_file config/evaluate.yml --snapshot data/snapshots/minivgg_lr0.0006_bs128_sgd3_wd0.00010_do0.55_sdo0.03_fb64_11-0.17.h5`

Set **knn_compare** and **activation_maps** to True to get kNearestNeighbors performance and class activation maps for predicted images respectively.

### Test on a single image
`python test.py --snapshot data/snapshots/minivgg_lr0.0006_bs128_sgd3_wd0.00010_do0.55_sdo0.03_fb64_11-0.17.h5 --config_file config/test.yml --image data/logs/test.jpg`
