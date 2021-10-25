import os
from os import path
import sys
from pprint import pprint # Used to print config nicely
import re

import numpy as np

# Import local files
import argparsing as ap
import dataloader as dl
import model as m
import visualization as vis


def stdout_to_file_setup(args):
    """
    Sends the stdout output to a file instead of displaying it on-screen.
    This is very helpful for keeping track of past runs and what hyperparameters
    they were using!
    """
    
    # Find a log number - smallest possible which is 0 and above
    logs = sorted([f for f in os.listdir(args.output_path) if 'log' in f])
    if len(logs) == 0:
        num = 0
    else:
        logs = [int(re.findall(r'\d+', l)[0]) for l in logs]
        logs.sort()
        found = False
        for i in range(len(logs)):
            if i != logs[i]:
                found = True
                num = i
                break
        
        if not found:
            num = len(logs)
    
    stdout_origin=sys.stdout 
    sys.stdout = open(path.join(args.output_path, "log{}.txt".format(num)), "w")
    suffix = str(num)
    return stdout_origin, suffix

    



if __name__ == "__main__":
    
    # Parse the arguments
    args = ap.parse_arguments()

    # Define a config of hyperparameters
    config = {
        # Dataloader hyperparameters
        'batch_size': args.batch_size,
        
        # Model/optimizer hyperparameters
        'lr': args.lr,
        'lr_decay_gamma': args.lr_decay_gamma,
        
        # Training hyperparameters
        'epochs': args.epochs,
        'early_stopping_patience': args.patience,
    }

    # Setup logging stdout to file
    if args.stdout_to_file:
        stdout_origin, suffix = stdout_to_file_setup(args)
    else:
        suffix = None
    
    # Print the current task being executed for future reference
    print("Task: {}".format(args.task))
    
    # Define the dataloaders
    train_loader, val_loader, test_loader = \
        dl.prepare_dataloaders(args.data_path, 15000, 1000, \
        args.load_in_batches, config['batch_size'])

    class_letters = list("ABCDEFGHIJ")
    class_to_letter = dict(zip(range(len(class_letters)), class_letters))

    # Visualize a random image and print its class
    if args.visualize_images:
        # Requires on-screen output, and for the data to be loaded
        assert args.load_in_batches is False and args.stdout_to_file is False
        vis.visualize_random_images(train_loader, class_to_letter)

    # Get a device (GPU if possible, otherwise CPU)
    device = m.get_device()
    
    # Print the hyperparameter config for future reference
    print("\nConfig:")
    pprint(config)
    print("\n")

    # Train
    histories = []
    models = []
    for _ in range(args.train_runs):
        # Define model
        model = m.Model(args.task, 28*28, len(class_letters), config, device)

        # Train the model
        history = model.train(train_loader, val_loader, config, device, verbose=args.verbose)
        print("Training took {:.3f} minutes".format(sum(history['times'])/60))
        
        if args.test:
            test_loss, test_acc = model.get_loss(test_loader, device)
            print("Test loss:", np.round(test_loss, 5))
            print("Test accuracy:", np.round(test_acc, 5))
        
        histories.append(history)
        models.append(model)
        print("\n"*3)
        
    # Plot histories
    vis.plot_histories(histories, save_path=args.output_path, suffix=suffix, show_mean=False)
    
    
    # Close the stdout pipe if stdout was being sent to a file
    if args.stdout_to_file:
        sys.stdout.close()
        sys.stdout=stdout_origin