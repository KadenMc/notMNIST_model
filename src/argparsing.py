import os
import argparse

# Define relative paths
SRC_PATH = str(os.path.dirname(os.path.abspath(__file__)))
try:
    ROOT_PATH = SRC_PATH[:SRC_PATH.rindex('/')]
except:
    ROOT_PATH = SRC_PATH[:SRC_PATH.rindex('\\')]
DATA_PATH = os.path.join(ROOT_PATH, 'data')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'outputs')

def path(path):
    if os.path.isdir(path) or os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path".format(path))


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid file path".format(path))


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid directory path".format(path))


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Paths arguments
    parser.add_argument('--data_path', type=dir_path, default=DATA_PATH, \
        help='Path to save model history')
    parser.add_argument('--output_path', type=dir_path, default=OUTPUT_PATH, \
        help='Path to save model history')

    # Task arguments
    parser.add_argument("--task", type=str, default='5', \
        help="Specify the task to run out of options: '2', '3-100', '3-500', '3-1000', '4', and '5' - defaults to 5")
    
    # Logging arguments
    parser.add_argument("--stdout_to_file", action='store_true', \
        help="If flagged, log standard output to file in output path")
    
    parser.add_argument("--verbose", action='store_true', \
        help="Specify training verbosity")
    
    # Training & testing arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9, help="Learning rate StepLR decay gamma")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    parser.add_argument("--train_runs", type=int, default=1, help="Number of training runs")
    
    parser.add_argument("--test", action='store_true', help="Training verbosity")
    
    # Data loading arguments
    parser.add_argument("--load_in_batches", action='store_true', \
        help="If flagged, load the data in batches, otherwise load all - always trains in batches")
    
    # Visualization arguments
    parser.add_argument("--visualize_images", action='store_true', \
        help="Visualize random images from the dataset")
    
    args = parser.parse_args()
    return args
