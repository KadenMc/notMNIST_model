from os.path import join
from tqdm import tqdm # Creates custom progress bar
import time # For timing training

# Numpy & PyTorch imports
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# Imports for model saving & loading
from argparsing import MODELS_PATH
from os.path import join

def get_device(verbose=True):
    """
    Get the device on which to train.
    Use a GPU if possible, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        if device.type == 'cuda':
            print("Using Device:", torch.cuda.get_device_name(0))
        else:
            print("Using Device:", device)
    
    return device

def exit_training():
    from argparsing import ROOT_PATH
    f = open(join(ROOT_PATH, 'exit.txt'), 'r')
    exit_training = int(f.read())
    f.close()
    
    # If manually exiting, reset the file to have text "1"
    # so this doesn't have to be changed back manually
    if exit_training:
        f = open(join(ROOT_PATH, 'exit.txt'), 'w')
        f.write("0")
        f.close()
    
    return exit_training
    

class MyEarlyStopping:
    """
    A custom early stopping class based on a patience hyperparameter.
    If we are 'patience' epochs without improvement in our metric,
    then end the training. Note that the metric is provided, not
    hard-coded.
    """
    def __init__(self, patience):
        self.history = np.array([])
        self.patience = patience
    
    def step(self, metric, mode=np.argmin):
        self.history = np.append(self.history, metric)
        last_ind_improved = len(self.history) - (mode(self.history) + 1)
        if last_ind_improved > self.patience:
            return True
        return False


class Model:
    def __init__(self, config, device, print_summary=True):
        # Define the model
        self.model = nn.Sequential(
            # (None, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # (None, 32, 28, 28)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 14, 14)
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (None, 64, 14, 14)
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 7, 7)
            nn.Dropout2d(p=0.25),
            nn.Flatten(),
            nn.Linear(64*7*7, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
        
        if print_summary:
            print(self.model)

        # Send model to device
        self.model.to(device)
        
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        
        # Define learning rate scheduler (handles decay)
        if config['lr_decay_gamma'] is not None:
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=config['lr_decay_gamma'])
        else:
            # gamma = 1 is the same as having no decay at all
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=1)
    
        self.early_stopping = MyEarlyStopping(config['early_stopping_patience'])
    
    def get_loss(self, loader, device):
        """
        Gets a loss and accuracy given the current model and a dataloader
        from which to pull data.
        """
        accuracy = 0
        loss = 0
        for images, labels in loader:
            # Send batch to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Add to loss and accuracy
            probs = self.model(images)
            loss += self.loss_fn(probs, labels).item()
            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return loss/len(loader), accuracy/len(loader)
    
    def predict(self, model_path, loader, device):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        return self.get_loss(loader, device)
    
    def train(self, train_loader, val_loader, config, device, verbose=True, \
        tensorboard_path=None, model_save_path=None):
        """
        Trains the model.
        """ 
        # Setup Tensorboard
        if tensorboard_path is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_path)
        
        # Train the model
        train_losses = []
        val_losses = []
        val_accuracy = []
        times = []
        for e in range(config['epochs']):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {e}")
                t_prev = time.time()
                batch_loss = 0
                for images, labels in tepoch:
                    # Send batch to device
                    images = images.to(device)
                    labels = labels.to(device)

                    # Training pass
                    self.optimizer.zero_grad()
                    
                    output = self.model(images)
                    loss = self.loss_fn(output, labels)
                    loss.backward()
                    self.optimizer.step()

                    batch_loss += loss.item()

                    tepoch.set_postfix(loss=loss.item())

                # Log all the stats
                times.append(time.time() - t_prev)
                train_losses.append(batch_loss/len(train_loader))
                val_loss, val_acc = self.get_loss(val_loader, device)
                val_losses.append(val_loss)
                val_accuracy.append(val_acc)
                
                # Print the stats if verbose is true
                if verbose:
                    print("Epoch {} Train Loss: {}".format(e, np.round(train_losses[-1], 5)))
                    print("Epoch {} Validation Loss: {}".format(e, np.round(val_losses[-1], 5)))
                    print("Epoch {} Val Accuracy: {}".format(e, np.round(val_accuracy[-1], 5)))
                    print("Epoch {} Time: {}".format(e, np.round(times[-1], 5)))
                    
                    if config['lr_decay_gamma'] is not None:
                        print("Epoch {} lr: {}".format(e, self.optimizer.param_groups[0]['lr']))
                
                # Write to TensorBoard
                if tensorboard_path is not None:
                    writer.add_scalar("Training Loss", train_losses[-1], e)
                    writer.add_scalar("Validation Loss", val_losses[-1], e)
                    writer.add_scalar("Validation Accuracy", val_accuracy[-1], e)
                
                t_prev = time.time()
                
                # Update the learning rate
                self.scheduler.step()
                
                # Check for early stopping
                if self.early_stopping.step(val_losses[-1]):
                    print("Stopped early - No val acc improvement in {} epochs".format( \
                        config['early_stopping_patience']))
                    break
                    
                # Check whether the training is to be stopped manually
                if exit_training():
                    print("Manually exiting training")
                    break
                
                # Save model on lowest validation loss
                if model_save_path is not None:
                    if val_losses[-1] == min(val_losses):
                        torch.save(self.model.state_dict(), model_save_path)
                        print("Saved model at epoch {} with val loss {}".format(e, val_losses[-1]))

                print()
        
        # Prepare and return the history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracy': val_accuracy,
            'times': times,
        }
        return history