# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:44:27 2022

@author: sergio
"""
import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from tensorflow.keras.utils import to_categorical
from  tqdm import  tqdm
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from tensorflow import keras 
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from visualize_results import visualize_results
import matplotlib

def testeo_patches(x_test,y_test,path_model,limite_pred):
    
    
    model = keras.models.load_model(path_model)     # we load the model
            
    val_preds = model.predict(x_test)                # we predict the labels of the input patches


    etiquetas_pred_porcentajes = val_preds[:,1]
    etiquetas_pred_porcentaje_mas_porc_fijado = val_preds[:,1]

    val_preds = to_categorical(np.argmax(val_preds, axis=1),num_classes=2)

    etiquetas_pred_binarias=val_preds.argmax(axis=1)
    etiquetas_reales=y_test.argmax(axis=1)




    porct=limite_pred
    etiquetas_pred_porcentaje_fijado=[]
    ett=[]
    
    for i in range(len(etiquetas_pred_porcentaje_mas_porc_fijado)):
        if etiquetas_pred_porcentaje_mas_porc_fijado[i] < (porct/100) :
            etiquetas_pred_porcentaje_fijado.append(0)
            ett.append(0)
        else:
            etiquetas_pred_porcentaje_fijado.append(etiquetas_pred_porcentaje_mas_porc_fijado[i])
            ett.append(1)
        
    etiquetas_pred_binarias_porcentaje_fijado=np.array(ett)

    
    
    return etiquetas_reales,etiquetas_pred_binarias,etiquetas_pred_porcentajes,etiquetas_pred_porcentaje_fijado,etiquetas_pred_binarias_porcentaje_fijado
