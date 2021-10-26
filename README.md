# notMNIST

A fun PyTorch implementation of a deep neural network for predicting images from the `notMNIST_small` dataset.

The network uses 2 sets of layers with `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `Dropout2d`, and finally ends with two `Linear` layers. Training is performed with optimizer `AdamW` and a slowly decaying learning rate using `StepLR`.

The network is capable of achieving 95% on the validation set and 94% on the test set in just 12 epochs.

## How to Run
- Download the repository.
- Download the notMNIST data: https://www.kaggle.com/lubaroli/notmnist/version/1
- Unzip notMNIST_small and move the notMNIST letter folders directly into the data folder.
- Train the model, or download the pre-trained model below!

Utilize the commands and optional parameters below. Note that plots are always saved to the output folder as images.

## Commands

Terminal train command:

```
python main.py --verbose --test --stdout_to_file --use_tensorboard
```
*Note: The output will be sent to a log file in the output directory*


Then, in a Jupyter notebook,
```
!pip install tensorboard
```
```
%reload_ext tensorboard
%tensorboard --logdir=".../notMNIST/outputs/log0_run/" --reload_multifile True
```

Terminal predict command:

```
python main.py --predict --model_path ".../notMNIST/models/model0_run.pt"
```

## Argument Parsing

### Optional Parameters
- `--data_path`: Specify the data folder path. Defaults to `notMNIST/data` folder.
- `--model_path`: Path from which to load model parameters. Must be specified if `--predict` flagged.
- `--train_runs`: Specify the number of runs, or models to train.
- `--batch_size`: Specify training batch size.
- `--lr`: Specify learning rate.
- `--lr_decay_gamma`: Specify `torch.optim.lr_scheduler.StepLR` gamma.
- `--epochs`: Specify maximum number of epochs.
- `--patience`: Specify early stopping patience.



### Flags
- `--predict`: If flagged, predict, otherwise train.
- `--stdout_to_file`: Send stdout go to a log file in the outputs folder. The only exception is that the tqdm progress bar will still appear in console.
- `--verbose`: Specify training verbosity.
- `--use_tensorboard`: Setups up TensorBoard, which can be used to visualize training live.
- `--test`: Test after training the model.
- `--load_in_batches`: Load the data in batches, rather than all at once. Recommended to not flag unless having out of memory problems, as training is much faster.
- `--visualize_images`: Visualize random images from the training set. Cannot flag `--stdout_to_file` or `--load_in_batches` when visualizing images.

## Pre-trained Model

A pre-trained model can be downloaded [here](https://www.dropbox.com/s/gmixvawajx66snd/model0_run_downloaded.pt?dl=0) which will work with the existing architecture:

## Fun Features
- A `tqdm` training progress bar.
- A manual exit option using the `exit.txt` file. Save document with "1" to end the training, or "0" otherwise. Upon exiting, the program resets `exit.txt` to 0.
- Capability to train and visualize several runs with the same hyperparameters, as well as plot the 'average run'.