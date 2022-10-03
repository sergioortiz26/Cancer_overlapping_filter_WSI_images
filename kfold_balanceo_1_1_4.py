# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:10:17 2022

@author: sergio
"""

import cv2
import pandas as pd
import numpy as np 
import random 
import glob 
from tqdm import tqdm
from balanceo_patches import balanceo_parches
from balanceo_parches_ajuste_tumor import balanceo_parches_ajuste_tumor
import parameters 

path_patches_test_sin_overlapping='C:/Users/Sergio/Documents/CANCER/Patch_100x100_test_sin_overlapping/'
path_patches_test_con_overlapping='C:/Users/Sergio/Documents/CANCER/Patch_100x100_test_con_overlapping/'
patch_100x100_train_and_validation='C:/Users/Sergio/Documents/CANCER/Patch_100x100_train_and_validation/'


path_imagenes='C:/Users/Sergio/Documents/CANCER/HUP_images/'
folder_output='C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/inputs_train_test_validation3/'



def kfold(path_imagenes,patch_100x100_train_and_validation,path_patches_test_sin_overlapping,path_patches_test_con_overlapping,splits=5):  

    """Creating a k fold cressvalidation dataset"""

    paths=[]
    paths= glob.glob(path_imagenes+'/*.png')  # obtengo una lista con los path
    random.shuffle(paths)    #baraja 'shuffle' los paths

# One time that I had shuffled the images, I select 20% to test and the rest (80%) to train and validation

    Imagen_id=[]

    for path in paths:

        path=path.replace('\\','/' )

        filename = path.split('/')[-1].split('.')[0]
        filename = filename.split('_')[0]


        id_ = filename

        Imagen_id.append(id_)


    window = int(len(Imagen_id) /splits)


    for i in tqdm(range(splits)) :   # de 0 a 4  divido los datos de train en 5 -> 1 para validacion y 4 realmente para train

        start = window*i
        end = window*(i+1)
        Imagenes_test = Imagen_id[start:end]        
        Imagenes_val = Imagen_id[start:end]

        Imagenes_train = []

        for j in range (splits) :
            # if it is the test split

            start_j =window*j
            end_j = window*(j+1)

            if(j!=i):

                Imagenes_train.append(Imagen_id[ start_j : end_j])


        Imagenes_train = [item for sublist in Imagenes_train for item in sublist]


        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test=[]
        y_test=[]    
        x_test2=[]
        y_test2=[] 

        patches_test_sin_overlapping=[]
        patches_test_con_overlapping=[]
        patches_val_sin_overlapping=[]
        patches_train_sin_overlapping=[]


        for j in range(len(Imagenes_test)):
            patches_test_sin_overlapping.extend(glob.glob(path_patches_test_sin_overlapping+Imagenes_test[j]+'*.png'))

        for j in range(len(Imagenes_test)):
            patches_test_con_overlapping.extend(glob.glob(path_patches_test_con_overlapping+Imagenes_test[j]+'*.png'))

        for j in range(len(Imagenes_val)):
            patches_val_sin_overlapping.extend(glob.glob(patch_100x100_train_and_validation+Imagenes_val[j]+'*.png'))

        for j in range(len(Imagenes_train)):
            patches_train_sin_overlapping.extend(glob.glob(patch_100x100_train_and_validation+Imagenes_train[j]+'*.png'))


        patches_train_sin_overlapping2=balanceo_parches_ajuste_tumor(patches_train_sin_overlapping)
        patches_train_sin_overlapping3=list(patches_train_sin_overlapping2.iloc[:,-1])
        random.shuffle(patches_train_sin_overlapping3)    #baraja 'shuffle' los paths

        patches_val_sin_overlapping2=balanceo_parches_ajuste_tumor(patches_val_sin_overlapping)
        patches_val_sin_overlapping3=list(patches_val_sin_overlapping2.iloc[:,-1])
        random.shuffle(patches_val_sin_overlapping3)    #baraja 'shuffle' los paths        
        
        
        
        for path in tqdm(patches_test_sin_overlapping):    #recorro todos los path de las imagenes de test
    
            img_ = cv2.imread(path)                #lee la imagen 
            img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
            x_test.append(img_rgb)
            
            f=path.replace('\\','/' )
            save_name = f.split('/')[-1]   #split()método divide una cadena en una lista
            etiqueta=save_name.split('.')[3]
            y_test.append(etiqueta)        
    
        print( "Number of samples in test within overlapping: {}".format(len(x_test))) 
    
    
            #Creamos un diccionario donde tenemos poir un lado un array con los datos de las imagenes y por otro sus etiquetas
        dict_split = {
            'x_test':np.asarray(x_test) ,
            'y_test':np.asarray(y_test),
            'Name_imagen':np.asarray(Imagenes_test)
            }
    
            #Guardamos un diccionario con las imagenes x y sus etiquetas de test
        np.save(folder_output+'inputs_labels_split_path_test_sin_overlapping_split_'+str(i)+'_'+ str(parameters.height)+ "x" +str(parameters.width)+'.npy',dict_split)
        
        
        for path in tqdm(patches_test_con_overlapping):    #recorro todos los path de las imagenes de test
    
            img_ = cv2.imread(path)                #lee la imagen 
            img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
            x_test2.append(img_rgb)
            
            f=path.replace('\\','/' )
            save_name = f.split('/')[-1]   #split()método divide una cadena en una lista
            etiqueta=save_name.split('.')[3]
            y_test2.append(etiqueta)        
    
        print( "Number of samples in test with overlapping: {}".format(len(x_test2))) 
    
    
            #Creamos un diccionario donde tenemos poir un lado un array con los datos de las imagenes y por otro sus etiquetas
        dict_split = {
            'x_test':np.asarray(x_test2) ,
            'y_test':np.asarray(y_test2),
            'Name_imagen':np.asarray(Imagenes_test)
            }
    
            #Guardamos un diccionario con las imagenes x y sus etiquetas de test
        np.save(folder_output+'inputs_labels_split_path_test_con_overlapping_split_'+str(i)+'_'+ str(parameters.height)+ "x" +str(parameters.width)+'.npy',dict_split)
    
    
        for path in tqdm(patches_val_sin_overlapping3):    #recorro todos los path de las imagenes de test
    
            img_ = cv2.imread(path)                #lee la imagen 
            img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
            x_val.append(img_rgb)
            
            f=path.replace('\\','/' )
            save_name = f.split('/')[-1]   #split()método divide una cadena en una lista
            etiqueta=save_name.split('.')[3]
            y_val.append(etiqueta)        
    
        print( "Number of samples in validation: {}".format(len(x_val))) 
        
        
        for path in tqdm(patches_train_sin_overlapping3):    #recorro todos los path de las imagenes de test
    
            img_ = cv2.imread(path)                #lee la imagen 
            img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
            x_train.append(img_rgb)
            
            f=path.replace('\\','/' )
            save_name = f.split('/')[-1]   #split()método divide una cadena en una lista
            etiqueta=save_name.split('.')[3]
            y_train.append(etiqueta)        
    
        print( "Number of samples in train: {}".format(len(x_train))) 
        
        
    
            #Creamos un diccionario donde tenemos poir un lado un array con los datos de las imagenes y por otro sus etiquetas
        dict_split = {
            'x_train':np.asarray(x_train) ,
            'y_train':np.asarray(y_train),
            'x_val':np.asarray(x_val) ,
            'y_val':np.asarray(y_val)
        }
    
            #Guardamos un diccionario con las imagenes x y sus etiquetas de test
            #np.save('../patches/inputs/inputs_labels_split_patch_' + str(height)+ "x" +str(width) + '.npy',dict_split)
        np.save(folder_output+'inputs_labels_split_path_train_validation_split_split_'+str(i)+'_'+ str(parameters.height)+ "x" +str(parameters.width)+'.npy',dict_split)


# dividimos en 5 cogemos 1 para test y 4 para train 
#los de test los guardamos y los de train los volvemos a dividir en 5, 4 para el train propiamente dichi y 1 para validacion