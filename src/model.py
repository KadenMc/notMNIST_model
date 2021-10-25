from tqdm import tqdm # Creates custom progress bar
import time # For timing training

# Numpy & PyTorch imports
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

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

def get_sequential_model(task, input_size, output_size):
    """
    Select and return the correct model for the given task.
    """
    # TASK 2
    if task == "2":
        model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )
    
    # TASK 3
    elif task == "3-100":
        # 100
        model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )
    
    # 500
    elif task == "3-500":
        model = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )
    
    # 1000
    elif task == "3-1000":
        model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )

    # TASK 4
    elif task == "4":
        model = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )
    
    # TASK 5
    elif task == "5":
        model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, output_size),
            #nn.Softmax(dim=1) # Unnecessary is using nn.CrossEntropyLoss()
        )
    
    return model


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
    def __init__(self, task, input_size, output_size, config, device, print_summary=True):
        # Define the model
        self.model = get_sequential_model(task, input_size, output_size)
        
        # Send model to device
        self.model.to(device)
        
        # Print a summary
        if print_summary:
            from torchsummary import summary
            summary(self.model, (input_size,), device=device.type)
        
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'])
        
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
            # Get image predictions (taking class with max probability)
            images = images.view(images.shape[0], -1)
            
            # Send batch to device
            images = images.to(device)
            labels = labels.to(device)
            
            probs = self.model(images)
            loss += self.loss_fn(probs, labels).item()
            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return loss/len(loader), accuracy/len(loader)
    
    
    def train(self, train_loader, val_loader, config, device, verbose=True):
        """
        Trains the model.
        """  
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
                    # Flatten
                    images = images.view(images.shape[0], -1)
                
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
                
                    print()
                
                t_prev = time.time()
                
                # Update the learning rate
                self.scheduler.step()
                
                # Check for early stopping
                if self.early_stopping.step(val_losses[-1]):
                    print("Stopped early - No val acc improvement in {} epochs".format( \
                        config['early_stopping_patience']))
                    break
        
        # Prepare and return the history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracy': val_accuracy,
            'times': times,
        }
        return history