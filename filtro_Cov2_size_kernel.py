# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:06:53 2022

@author: sergio
"""

# example of using a single convolutional layer
import numpy as np
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
# define input data

# create model

def filtro_Cov2_size_kernel(image,size_kernel,stride_x_y,kernel):
    
    model = Sequential()
    model.add(Conv2D(1, size_kernel ,strides= stride_x_y, input_shape=image.shape[1:]))
    # summarize model
    #model.summary()
    # define a vertical line detector
        
    weights = [asarray(kernel), asarray([0.0])]
    # store the weights in the model
    model.set_weights(weights)
    # apply filter to input data
    output_predicciones = model.predict(image)
    output_predicciones = np.squeeze(output_predicciones)
    output_binario=np.round(output_predicciones)
    
    return output_predicciones, output_binario
