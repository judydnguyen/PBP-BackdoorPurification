# This is the implementation of PBP: Post-training Backdoor Purification for Malware Classification

## Dataset
The pre-processed datasets for AndroZoo is stored at `datasets/apg/`. \
Due to the size restriction, we encourage the community to download and extract EMBER dataset followed the instruction of the original paper at [Explanation-guided Backdoor Attacks](https://github.com/ClonedOne/MalwareBackdoors).

After you can achieve the extracted EMBER data, put the four files `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat` into `datasets/ember` folder.

## Requirements
Here is the packages required to reproduce our results
```
>>> torch.__version__
'2.1.0+cu121'
Python 3.10.12
```
Other packages please refer at `requirements.txt`
## How to start fine-tuning a backdoored model
We provide checkpoints to reproduce  the results in Table III
The checkpoints are the weight of backdoored models with AndroZoo stored at `models/apg/torch/embernn/backdoor`
```
chmod +x ./run_baselines_jigsaw.sh &&
./run_baselines_jigsaw.sh
```
The corresponding logging results will be stored at 

## How to start training a backdoor model:
```
chmod +x ./train_models.sh
```
We provide both scripts for training backdoor for AndroZoo and EMBER datasets.

## Configs
The configs can be found in [this](configs/backdoors)

## Hyper-parameters
| Name                | Type      | Description                                                                                     | Default                          |
|---------------------|-----------|-------------------------------------------------------------------------------------------------|----------------------------------|
| `device`            | `str`     | The device to train on (e.g. `'cpu'`, `'cuda:0'`).                                              | `"cuda"`                         |
| `ft_mode`           | `str`     | Fine-tuning mode (`'all'`, `'fe-tuning'`, etc.).                                                | `"fe-tuning"`                    |
| `num_classes`       | `int`     | Number of classes in the dataset.                                                                | `25`                             |
| `attack_label_trans`| `str`     | The type of label transformation for the attack (`'all2one'`, `'one2one'`, etc.).                | `"all2one"`  |
| `pratio`            | `float`   | The poison ratio if applicable.                                                                  | `None`        |
| `epochs`            | `int`     | Number of training epochs.                                                                       | `10`                             |
| `dataset`           | `str`     | Name of the dataset to be used.                                                                  | `"malimg"`                       |
| `dataset_path`      | `str`     | Path to the dataset directory.                                                                   | `"datasets/malimg"`     |
| `folder_path`       | `str`     | Path to the folder where models are saved.                                                       | `"models/malimg/torch"` |
| `attack_target`     | `int`     | The target class for backdoor attacks.                                                           | `0`          |
| `batch_size`        | `int`     | Batch size for training.                                                                         | `64`                             |
| `test_batch_size`   | `int`     | Batch size for testing.                                                                          | `512`                            |
| `random_seed`       | `int`     | Random seed for reproducibility.                                                                 | `0`         |
| `model`             | `str`     | Model architecture to use.                                                                       | `"mobilenetv2"`                  |
| `split_ratio`       | `float`   | The ratio of the split between training and validation sets.                                     | `None`      |
| `log`               | `bool`    | Whether to log outputs to a file.                                                                | `False`     |
| `initlr`            | `float`   | Initial learning rate if using a learning rate scheduler.                                        | `None`      |
| `pre`               | `bool`    | Whether to pre-train the model.                                                                  | `False`     |
| `save`              | `bool`    | Whether to save the trained model.                                                               | `False`     |
| `linear_name`       | `str`     | Name of the linear layer (last layer), if different from the default.                                         | `"classifier"`                   |
| `lb_smooth`         | `float`   | Label smoothing value, if used.                                                                  | `None`      |
| `alpha`             | `float`   | Hyperparameter to balance loss terms (e.g. in feature shift regularization).                    | `0.2`       |
| `f_epochs`          | `int`     | Number of fine-tuning epochs.                                                                    | `5`                              |
| `f_lr`              | `float`   | Fine-tuning learning rate.                                                                       | `0.0002`                         |
| `imsize`            | `int`     | The size of the images for the model input.                                                      | `64`                             |
| `conv1`             | `int`     | Number of output channels for the first convolutional layer.                                     | `32`                             |
| `target_label`      | `int`     | Target label used in attacks.                                                                    | `0`                              |
| `is_backdoor`       | `int`     | Indicator whether the dataset contains backdoored samples (1 for yes, 0 for no).                 | `1`                              |
| `lr`                | `float`   | Learning rate for the optimizer.                                                                 | `0.002`                          |
| `ft_size`           | `float`   | Portion of the dataset to use for fine-tuning.                                                   | `0.05`                           |
| `num_poison`        | `int`     | Number of poisoned samples.                                                                      | `2`                              |
| `poison_rate`        | `float`     | Poisoning rate for training samples.                                                                      | `0.01`                              |
