from os import path

import numpy as np
import matplotlib.pyplot as plt
import cv2 # Only used to display images interactively


def visualize_random_images(train_loader, class_to_letter):
    """
    Visualizes random images from the training dataset.
    """
    X = train_loader.dataset.dataset.data.cpu().detach().numpy()
    y = train_loader.dataset.dataset.targets.cpu().detach().numpy()
    
    inp = None
    
    while inp != 'q':
        display_ind = np.random.choice(len(X), size=1, replace=False)
        cv2.imshow('img', np.squeeze(X[display_ind] * 255).astype(np.uint8))
        print("Class:", class_to_letter[y[display_ind][0]])
        cv2.waitKey(0)
        inp = input("Input 'q' to exit, or anything else to visualize more: ")


def save_image(save_path, key, suffix=None):
    """
    Saves an image to a given path.
    """
    if save_path is not None:
        if suffix is None:
            file = path.join(save_path, key) + '.png'
        else:
            file = path.join(save_path, key) + suffix + '.png'
        plt.savefig(file)


def plot_history(history, save_path=None, suffix=None, alpha=1):
    """
    Plots a model training history.
    """
    plt.rcParams["figure.figsize"] = (15, 5)
    
    for i, key in enumerate(history):
        plt.figure(i)
        plt.title(key)
        plt.plot(range(len(history[key])), history[key], alpha=alpha, color='black', lw=2)
        save_image(save_path, key, suffix=suffix)


def plot_mean(histories, key):
    """
    Plots the average of multiple training histories for a given metric, or key.
    """
    arrs = [np.array(h[key]) for h in histories]
    
    mean = np.array([])
    
    while len(arrs) > 0:
        shortest = min([len(a) for a in arrs])    
        temp = np.array([a[:shortest] for a in arrs]).mean(axis=0)
        mean = np.append(mean, temp)
        arrs = [a[shortest:] for a in arrs if len(a[shortest:]) > 0]

    print(key, mean)
    plt.plot(range(len(mean)), mean, color='black', lw=3)


def plot_histories(histories, save_path=None, suffix=None, show_mean=True):
    """
    Plots model training histories.
    """
    # If only one run, plot normally
    if len(histories) == 1:
        plot_history(histories[0], save_path=save_path, suffix=suffix)
        return

    # Otherwise, if show_mean is false, plot all runs normally,
    # if show_mean is true, then plot each run individually with
    # a lower alpha and then plot the average fully opaque
    alpha = 0.5 if show_mean else 1
    
    for h in histories:
        plot_history(h, alpha=alpha)

    # Plot the mean line and save each plot
    for i, key in enumerate(histories[0]):
        plt.figure(i)
        
        if show_mean:
            plot_mean(histories, key)
        save_image(save_path, key, suffix=suffix)