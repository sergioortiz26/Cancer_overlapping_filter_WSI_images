# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:39:38 2022

@author: sergio
"""

import matplotlib.pyplot as plt

def visualize_results(history,name):
            
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    
    plt.plot(epochs, acc, 'b', label='Training acc')        
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(name)
    
    