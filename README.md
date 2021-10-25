# notMNIST

## Notes
Plots are always saved to the output folder as images.


## Terminal Commands

*Run the program - output will be sent to a log file in the output directory*

`python main.py --task 5 --verbose --test --stdout_to_file`

*Visualize the images - Trains afterwards*

`python main.py --task 5 --verbose --test --visualize_images`




## Argument Parsing

### Optional Parameters

- `--task`: Determines which model which will be run from the assignment. Options: 2, 3-100, 3-500, 3-1000, 4, and 5. Defaults to 5.
- `--data_path`: Specify the data folder path. Defaults to `notMNIST/data`.
- `--output_path`:Specify the output folder path. Defaults to `notMNIST/outputs`.
- `--train_runs`: Specify the number of runs, or models to train.
- `--batch_size`: Specify training batch size.
- `--lr`: Specify learning rate.
- `--lr_decay_gamma`: Specify `torch.optim.lr_scheduler.StepLR` gamma.
- `--epochs`: Specify maximum number of epochs.
- `--patience`: Specify early stopping patience.



### Flags
- `--stdout_to_file`: Send stdout go to a log file in the outputs folder. The only exception is that the tqdm progress bar will still appear in console.
- `--verbose`: Specify training verbosity.
- `--test`: Test after training the model.
- `--load_in_batches`: Load the data in batches, rather than all at once. Recommended to avoid flagging unless having out of memory problems.
- `--visualize_images`: Visualize random images from the training set. You cannot flag "--stdout_to_file" or "--load_in_batches" when you want to visualize images.