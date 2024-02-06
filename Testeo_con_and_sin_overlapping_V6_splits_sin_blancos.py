
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:04:29 2022

@author: sergio
"""



# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:00:44 2022

@author: sergio
"""

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
from visualize_results import visualize_results
import matplotlib
from plot_confusion_matrix import plot_confusion_matrix
from visualize_results import visualize_results
import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from filtro_Cov2_size_kernel import filtro_Cov2_size_kernel
from data_reader import data_reader
from utils import data_of_imagen_in_question,testeo_patches
import os
from elimina_parches_blancos_etiquetas import elimina_parches_blancos_etiquetas
from organiza_datos import organiza_datos
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef


# ---------------------------------------------------- VARIABLE INITIALISATION

#import the model that we saved previously with the script CNN_V2_s2_balanceo.py

#carpeta_modelo_concreto='DenseNet121_ADAM_trainable_20%'
carpeta_modelo_concreto='DenseNet121_SGD_trainable_70%'


#path_model='C:/Users/Sergio/Documents/CANCER/script/model_ResNet50_50_epochs_256_batchsize_Adam_trainable_20%_optimiz_100x100_inputs_labels_split_path_0-100x100_rgb_15_porc_sin_14_test.h5'
#path_file_model='C:/Users/Sergio/Documents/CANCER/modelos/'

path_file_model='C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/modelos3/'
#path_model='C:/Users/Sergio/Documents/CANCER/script/model_VGG16_50_epochs_256_batchsize_ADAM_optimiz_100x100_inputs_labels_split_path_0-100x100_rgb_15_porc_sin_14_test.h5'

#path_imagenes='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_imgs_idx5/CINJ_imgs_idx5/'
path_imagenes='C:/Users/Sergio/Documents/CANCER/HUP_images/'


#path_inputs_patches='C:/Users/Sergio/Documents/CANCER/testeo_con_imagenes_CINJ/inputs_test/'
path_inputs_patches='C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/inputs_train_test_validation3/'

path_output_test='C:/Users/Sergio/Desktop/'

#path_mascaras='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_masks_HG/CINJ_masks_HG/'
path_mascaras='C:/Users/Sergio/Documents/CANCER/HUP_masks/'


# path_imagenes='C:/Users/Sergio/Documents/CANCER/HUP_images/'
# path_inputs_patches='C:/Users/Sergio/Documents/CANCER/balanceo/inputs_train_test_validation/'
# path_mascaras='C:/Users/Sergio/Documents/CANCER/HUP_masks/'


version='_version_70%_V1'


classes = ['tumor', 'normal']
classes2=['normal','tumor']


limite_pred=70
splits=5


# ---------------------------------------------------- READING OF THE PATCHES FROM THE TEST IMAGES WITHOUT OVERLAPPING

#nombre_imagenes_test=['12880','12886','8863','8864','8865','8867']


etiquetas_reales_total=[]
etiquetas_pred_binarias_total=[]
etiquetas_pred_binarias_porcentaje_fijado_total=[]
etiquetas_reales2_total=[]
etiquetas_pred_binarias_porcentaje_fijado2_total=[]
etiquetas_pred_binarias2_total=[]
labels_real_2_total=[]

label_preds_overlapping_total=[]
label_preds_overlapping_2_total=[]
label_preds_overlapping_3_total=[]
label_preds_overlapping_4_total=[]
label_preds_overlapping_5_total=[]
label_preds_overlapping_6_total=[]
label_preds_overlapping_7_total=[]
label_preds_overlapping_8_total=[]
    
    
# directorio = path_output_test+'DenseNet121_split_50_epochs_256_batchsize_SGD_trainable_70%_optimiz_100x100_inputs_labels_split_path_0-100x100/'

# try:
#     os.stat(directorio)
# except:
#     os.mkdir(directorio)    
    

tablas_normal={}
tablas_tumor={}
tablas_accuracy={}
tablas_BACC={}
tablas_MCC={}


for split in range(splits):
    
    
    name_model='model_DenseNet121_split_'+str(split)+'_100_epochs_256_batchsize_SGD_trainable_70%_optimiz_100x100_inputs_labels_split_path_0-100x100.h5'
    path_model= path_file_model+name_model  


    name_model=name_model.split('.')[0]


    Data_sin_overlapping=data_reader('DATA split '+str(split)+' imagenes test sin overlapping')   #I create a data object

    path_patches_sin_overlapping=path_inputs_patches+'inputs_labels_split_path_test_sin_overlapping_split_'+str(split)+'_100x100.npy'
    Data_sin_overlapping.save_data(path_patches_sin_overlapping,'test',classes)    # I call the save data method of the object, to save the data (indicating what I want to read, in this case the 'test' data).

    #I get the label and image data. I also have this data associated to the name of the image.
    x_test_sin_overlapping = Data_sin_overlapping.data['test']['x_test']
    y_test_sin_overlapping = Data_sin_overlapping.data['test']['y_test']
    name_image_test=Data_sin_overlapping.data['test']['name_imges_test']

    nombre_imagenes_test=name_image_test    #view the output format of this variable and compare it to that of another tsteo script

    #    ---------------------------------------------------- READING PATCHES FROM OVERLAPPING TEST IMAGES
    
    #(This database has been previously created with the same images as above but with the overlapping process implemented).
    
    Data_con_overlapping=data_reader('data1 14 imagenes sin overlapping')
    
    path_patches_con_overlapping=path_inputs_patches+'inputs_labels_split_path_test_con_overlapping_split_'+str(split)+'_100x100.npy'
    Data_con_overlapping.save_data(path_patches_con_overlapping,'test',classes)
    
    x_test_con_overlapping = Data_con_overlapping.data['test']['x_test']
    y_test_con_overlapping = Data_con_overlapping.data['test']['y_test']
    name_image_test=Data_sin_overlapping.data['test']['name_imges_test']
    
    # We want to compare the effect of doing the overlay and applying the filter vs. just testing the images without the overlay.
    #Initialise the variables that we are going to use
    
    etiquetas_reales_sin_blanco_append=[]
    etiquetas_pred_binarias_append=[]
    etiquetas_pred_binarias_porcentaje_fijado_append=[]
    etiquetas_reales2_append=[]
    etiquetas_pred_binarias_porcentaje_fijado2_append=[]
    etiquetas_pred_binarias2_append=[]
    labels_real_2_append=[]
    
    label_preds_overlapping_append=[]
    label_preds_overlapping_2_append=[]
    label_preds_overlapping_3_append=[]
    label_preds_overlapping_4_append=[]
    label_preds_overlapping_5_append=[]
    label_preds_overlapping_6_append=[]
    label_preds_overlapping_7_append=[]
    label_preds_overlapping_8_append=[]

    
    w=0
    z=0              # z is the counter for the beginnings of the images without overlapping.
    z1=0            # z1 is the counter of the beginnings of the overlapping images.
    
    
    
    
    for w in range(len(nombre_imagenes_test)):
        
        
        
            normal=[]
            tumor=[]
            accuracy_array=[]
            BACC_array=[]
            MCC_array=[]
            
            normalfloat=np.array(normal, dtype = np.float32)
            tumorfloat=np.array(tumor, dtype = np.float32)
            accuracy_float=np.array(accuracy_array, dtype = np.float32)
            BACC_float=np.array(BACC_array, dtype = np.float32)
            MCC_float=np.array(MCC_array, dtype = np.float32)
           
            etiquetas_reales=[]
            etiquetas_pred_binarias=[]
            etiquetas_pred_porcentajes=[]
            etiquetas_pred_porcentaje_fijado=[]
            etiquetas_pred_binarias_porcentaje_fijado=[]  # fix an umbral in the percentage of prediction and transform to binari de prediction.
            
        
    # reading of the imagen in question that I am going to test
    
            img_bgr=cv2.imread(path_imagenes+nombre_imagenes_test[w]+'_idx5.png')
            img_ = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
            #matplotlib.pyplot.imshow(img_rgb)
        
            
        
    ###### -------------------------------------------------------------------------------- IMAGENES SIN OVERLAPPING ------------------------------------------------------------------------------    
            # I select from all test image data, only those of the image in question.
        
            y_test_sin_overlapping_imagen_seleccionada,x_test_sin_overlapping_imagen_seleccionada,dimensions_image_sin_overlapping=data_of_imagen_in_question(
                                                                                                                                        img_,step_patch=100,position_data=z,y_test=y_test_sin_overlapping,x_test=x_test_sin_overlapping)
            
                            
            max_x=dimensions_image_sin_overlapping['max_x']
            max_y=dimensions_image_sin_overlapping['max_y']
    
            # For that image I predict the labels for each of its patches

            # test_patches is the function that loads the model and processes the data on the model to obtain the predictive image labels.
        
            [etiquetas_reales,etiquetas_pred_binarias,etiquetas_pred_porcentajes,etiquetas_pred_porcentaje_fijado,etiquetas_pred_binarias_porcentaje_fijado]=testeo_patches(
                                                                                                                                        x_test_sin_overlapping_imagen_seleccionada,y_test_sin_overlapping_imagen_seleccionada,path_model,limite_pred)
            
            #We need to remove the with patches
            etiquetas_pred_binarias_sin_blancos,etiquetas_reales_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, etiquetas_pred_binarias, etiquetas_reales)
            etiquetas_pred_binarias_porcentaje_fijado_sin_blancos,etiquetas_reales_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, etiquetas_pred_binarias_porcentaje_fijado, etiquetas_reales)
            
            etiquetas_reales_sin_blanco_append.extend(etiquetas_reales_sin_blancos)

            # representation of the results

            acc=accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)
            print('Accuracy salida binaria imagen 100x100 sin overlapping = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)
            print(classification_report(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos, target_names=classes2))
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)))
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)))
            print(cm)
            
    
            etiquetas_pred_binarias_append.extend(etiquetas_pred_binarias_sin_blancos)
            
            
            acc=accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)
            print('Accuracy salida binaria imagen 100x100 sin overlapping con un porcentaje de predicción de tumor mayor al' +str(limite_pred)+'% = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)
            print(classification_report(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos, target_names=classes2))
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)))
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)))
            print(cm)
            
            
            etiquetas_pred_binarias_porcentaje_fijado_append.extend(etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)        
    
            etiquetas_reales_matriz = [list(etiquetas_reales[i*max_x:i*max_x+max_x]) for i in range(max_y)]
            etiquetas_pred_binarias_matriz= [list(etiquetas_pred_binarias[i*max_x:i*max_x+max_x]) for i in range(max_y)]
            etiquetas_pred_porcentajes_matriz= [list(etiquetas_pred_porcentajes[i*max_x:i*max_x+max_x]) for i in range(max_y)]
            etiquetas_pred_porcentaje_fijado_matriz= [list(etiquetas_pred_porcentaje_fijado[i*max_x:i*max_x+max_x]) for i in range(max_y)] 
                    
            aa=np.array(etiquetas_reales_matriz)
            
            cc=np.array(etiquetas_pred_binarias_matriz)
    
            gg=np.array(etiquetas_pred_porcentajes_matriz)
            
            ii=np.array(etiquetas_pred_porcentaje_fijado_matriz)
            
            
            j=0
            i=0
            differencia=[]
            differencia2=[]
            for e1, e2 in zip(etiquetas_pred_binarias_matriz,etiquetas_reales_matriz):
                differencia=[]
                for ee1, ee2 in zip(e1,e2):
                    differencia.append(ee1 + ee2)
                differencia2.append(differencia)
                
            ee=np.array(differencia2)
                
            
            x=100
            
            img2_=img_[0:(max_y)*100,0:(max_x)*100,:]
            
            
            #MUY IMPORTANTE- CAMBIAR CUANDO SE UTILICE otra base de datos

            #img_mask=cv2.imread(path_mascaras+nombre_imagenes_test[w]+'.png')
            img_mask=cv2.imread(path_mascaras+nombre_imagenes_test[w]+'_annotation_mask.png')

            img1=img_mask[0:(max_y)*100,0:(max_x)*100,:]
            
            
            img3_=np.ones(((max_y)*100,(max_x)*100,3))
            
            
            
            
            for i in tqdm(range(max_y)):
                for j in range(max_x):
            
                    if etiquetas_pred_porcentaje_fijado_matriz[i][j]<0.7:
            
                        img3_[i*x:(i+1)*x,j*x:(j+1)*x,:]=img2_[i*x:(i+1)*x,j*x:(j+1)*x,:]
                    else:
            
                        img3_[i*x:(i+1)*x,j*x:(j+1)*x,0]=[255]
                        img3_[i*x:(i+1)*x,j*x:(j+1)*x,1]=[1-etiquetas_pred_porcentaje_fijado_matriz[i][j]*255]
                        img3_[i*x:(i+1)*x,j*x:(j+1)*x,2]=[0]
            
            
            etiquetas=[]      
            img3_=img3_.astype(np.uint8) 
            
            
            dst = cv2.addWeighteddst = cv2.addWeighted(img2_,0.7,img3_,0.3,0)     
            
            dst2 = cv2.addWeighteddst = cv2.addWeighted(img2_,0.7,img1,0.3,0)     
    
                
            matplotlib.pyplot.title('Imagen Original')    
            matplotlib.pyplot.imshow(img_)
            plt.show()
            matplotlib.pyplot.title('Mascara')    
            matplotlib.pyplot.imshow(img_mask)
            plt.show() 
            matplotlib.pyplot.title('Mascara parches')    
            matplotlib.pyplot.imshow(aa)
            plt.show()
            matplotlib.pyplot.title('Decision 50%')    
            matplotlib.pyplot.imshow(cc)
            plt.show()
            matplotlib.pyplot.title('% pred de CNN')    
            matplotlib.pyplot.imshow(gg)
            plt.colorbar()
            plt.show()
            matplotlib.pyplot.title('Decision > 70%')    
            matplotlib.pyplot.imshow(ii)
            plt.show()        
            matplotlib.pyplot.title('Dif entre real y 50%')    
            matplotlib.pyplot.imshow(ee)
            plt.show()
            matplotlib.pyplot.title('Mapa de calor de la pred del 70%')    
            matplotlib.pyplot.imshow(dst)
            plt.show()
            matplotlib.pyplot.imshow(dst2)
            plt.show()
    
    
    
    ###### -------------------------------------------------------------------------------- IMAGES WITH OVERLAPPING ---------------------------------------------------------------------------------------------------- 
    #repet the process but with images with overlapping
    
            y_test_con_overlapping_imagen_seleccionada,x_test_con_overlapping_imagen_seleccionada,dimensions_image_con_overlapping=data_of_imagen_in_question(
                                                                                                                                        img_,step_patch=50,position_data=z1,y_test=y_test_con_overlapping,x_test=x_test_con_overlapping)
            max_x2=dimensions_image_con_overlapping['max_x']
            max_y2=dimensions_image_con_overlapping['max_y']
        
            max_x2=max_x2-1   #quit 1 due to in the creation patch con overlapping I had delete 1 position in each dimension, and for this now whem I am rebuilding the image I have in mind this 
            max_y2=max_y2-1         
            
            
            [etiquetas_reales2,etiquetas_pred_binarias2,etiquetas_pred_porcentajes2,etiquetas_pred_porcentaje_fijado2,etiquetas_pred_binarias_porcentaje_fijado2]=testeo_patches(x_test_con_overlapping_imagen_seleccionada,y_test_con_overlapping_imagen_seleccionada,path_model,limite_pred)
     
            etiquetas_pred_binarias2_sin_blancos,etiquetas_reales2_sin_blancos = elimina_parches_blancos_etiquetas(x_test_con_overlapping_imagen_seleccionada, etiquetas_pred_binarias2, etiquetas_reales2)
            etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos,etiquetas_reales2_sin_blancos = elimina_parches_blancos_etiquetas(x_test_con_overlapping_imagen_seleccionada, etiquetas_pred_binarias_porcentaje_fijado2, etiquetas_reales2)
                    
     
        
            etiquetas_reales2_append.extend(etiquetas_reales2_sin_blancos)
     
            acc=accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)
            print('Accuracy salida binaria imagen 100x100 con overlapping = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)
            print(classification_report(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos, target_names=classes2))
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)))
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)))
            print(cm)
             
            
            etiquetas_pred_binarias2_append.extend(etiquetas_pred_binarias2_sin_blancos) 
             
             
            acc=accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)
            print('Accuracy salida binaria imagen 100x100 con overlapping con un porcentaje de predicción de tumor mayor al' +str(limite_pred)+'% = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)
            print(classification_report(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos, target_names=classes2))
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)))
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)))
            print(cm)     
    
            etiquetas_pred_binarias_porcentaje_fijado2_append.extend(etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos) 
            
    
            etiquetas_reales2_matriz = [list(etiquetas_reales2[i*max_x2:i*max_x2+max_x2]) for i in range(max_y2)]
            etiquetas_pred_binarias2_matriz= [list(etiquetas_pred_binarias2[i*max_x2:i*max_x2+max_x2]) for i in range(max_y2)]
            etiquetas_pred_porcentajes2_matriz= [list(etiquetas_pred_porcentajes2[i*max_x2:i*max_x2+max_x2]) for i in range(max_y2)]
            etiquetas_pred_porcentaje_fijado2_matriz= [list(etiquetas_pred_porcentaje_fijado2[i*max_x2:i*max_x2+max_x2]) for i in range(max_y2)] 
            etiquetas_pred_binarias_porcentaje_fijado2_matriz= [list(etiquetas_pred_binarias_porcentaje_fijado2[i*max_x2:i*max_x2+max_x2]) for i in range(max_y2)] 
            
            
            ll=np.array(etiquetas_reales2_matriz)
            mm=np.array(etiquetas_pred_binarias2_matriz)
            nn=np.array(etiquetas_pred_porcentajes2_matriz)
            oo=np.array(etiquetas_pred_binarias_porcentaje_fijado2_matriz)
            
            matplotlib.pyplot.title('Overlapping Mascara parches')    
            matplotlib.pyplot.imshow(ll)
            plt.show()
            matplotlib.pyplot.title('Overlapping Decision 50%')    
            matplotlib.pyplot.imshow(mm)
            plt.show()
            matplotlib.pyplot.title('Overlapping % pred de CNN')    
            matplotlib.pyplot.imshow(nn)
            plt.colorbar()
            plt.show()
            
    
    #············ APPLY THE FILTERS ·······················

    ####################### FILTERS 3x3 with stride 2 ##########################
    
    #Filters on binary predictions 50%.
    
            img4_=np.ones(((max_y2)+2,(max_x2)+2))
            
            img4_[1:max_y2+1,1:max_x2+1]=mm
            
            img4_[0,1:max_x2+1]=mm[0,0:max_x2]
            img4_[max_y2+1,1:max_x2+1]=mm[max_y2-1,0:max_x2]
            
            img4_[1:max_y2+1,0]=mm[0:max_y2,0]
            img4_[1:max_y2+1,max_x2+1]=mm[0:max_y2,max_x2-1]
            
            img4_[0,0]=mm[0,0]
            img4_[0,max_x2+1]=mm[0,max_x2-1]
            img4_[max_y2+1,0]=mm[max_y2-1,0]
            img4_[max_y2+1,max_x2+1]=mm[max_y2-1,max_x2-1]
            
            image=img4_
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
            
            # define model containing just a single max pooling layer
            modelll = Sequential([AveragePooling2D(pool_size = 3, strides = 2)])
            
            # generate pooled output
            output = modelll.predict(image)
            output = np.squeeze(output)
            #output2=np.round(output)
            
            output2=np.copy(output)
            
            for i in range(output2.shape[0]):
                for j in range(output2.shape[1]):
                    
                    if output2[i][j]>=4/9:
                        output2[i][j]=1
                    else:
                        output2[i][j]=0 
    
    
    #Filters on binary predictions of % fixed
    
    
            img5_=np.ones(((max_y2)+2,(max_x2)+2))
            
            img5_[1:max_y2+1,1:max_x2+1]=oo
            
            img5_[0,1:max_x2+1]=oo[0,0:max_x2]
            img5_[max_y2+1,1:max_x2+1]=oo[max_y2-1,0:max_x2]
            
            img5_[1:max_y2+1,0]=oo[0:max_y2,0]
            img5_[1:max_y2+1,max_x2+1]=oo[0:max_y2,max_x2-1]
            
            img5_[0,0]=oo[0,0]
            img5_[0,max_x2+1]=oo[0,max_x2-1]
            img5_[max_y2+1,0]=oo[max_y2-1,0]
            img5_[max_y2+1,max_x2+1]=oo[max_y2-1,max_x2-1]
    
            image1=img5_
            image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1)
            
            # define model containing just a single max pooling layer
            modelll = Sequential(
                [AveragePooling2D(pool_size = 3, strides = 2)])
            
            # generate pooled output
            output3 = modelll.predict(image1)
            output3 = np.squeeze(output3)
            #output4=np.round(output3)
            
            output4=np.copy(output3)
            
            # We binarize the results to compar them with the True label
            for i in range(output4.shape[0]):
                for j in range(output4.shape[1]):
                    
                    if output4[i][j]>=4/9:   # fixed 4/9 because if 4 to 9 patch are tumor consider all with tumor (4 patches conform each corner of patch originalbinarizamos )
                        output4[i][j]=1
                    else:
                        output4[i][j]=0
                        
            
    ############ APPLY KERNEL ############
    
            kernel = [[[[1/16]],[[2/16]],[[1/16]]],
                      [[[2/16]],[[4/16]],[[2/16]]],
                      [[[1/16]],[[2/16]],[[1/16]]]]
            
            output13,output14=filtro_Cov2_size_kernel(image=image,size_kernel=(3,3),stride_x_y=(2,2),kernel=kernel)    #sobre las predicciones binarias 50%
            output15,output16=filtro_Cov2_size_kernel(image=image1,size_kernel=(3,3),stride_x_y=(2,2),kernel=kernel)   #sobre las predicciones binarias del % fijado
            
            
            
    
            
    ####################### FILTERS 2X2 with stride 1 + FILTER 2X2 with stride 2 ##########################
            
    # (50%)
    # make the filter 2x2 with stride 1 to calculate the average of each corner of the image
    
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 1)])
    
            # generate pooled output
            output5 = modelll.predict(image) # Use 'image' due to in this variable it is store the matrix prediction 'overlapping' with 50% (with padding)
            output6 = np.squeeze(output5)
    
    
            image2 = output6.reshape(1, output6.shape[0], output6.shape[1], 1)
    
    # make the filter 2x2 with stride 2 to calculate the average of the entire image
    
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 2)])
    
            # generate pooled output
            output7 = modelll.predict(image2)
            output7 = np.squeeze(output7)
            
            output8=np.copy(output7)
            
            for i in range(output8.shape[0]):
                for j in range(output8.shape[1]):
                    
                    if output8[i][j]>=1/4:
                        output8[i][j]=1
                    else:
                        output8[i][j]=0
                        
                        
            
                        
            
    # (% fixed)
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 1)])
    
            # generate pooled output
            output9 = modelll.predict(image1)  # Use 'image1' due to in this variable it is store the matrix prediction 'overlapping' with % fixed (with padding)
            output10 = np.squeeze(output9)
    
    
            image2 = output10.reshape(1, output6.shape[0], output6.shape[1], 1)
    
    
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 2)])
    
            # generate pooled output
            output11 = modelll.predict(image2)
            output11 = np.squeeze(output11)
            
            output12=np.copy(output11)
            
            for i in range(output12.shape[0]):
                for j in range(output12.shape[1]):
                    
                    if output12[i][j]>=1/4:
                        output12[i][j]=1
                    else:
                        output12[i][j]=0 
                        
    ############ APPLY KERNEL ############
                        
                        
            # kernel1 = [[[[1/9]],[[2/9]]],
            #           [[[2/9]],[[4/9]]]]
            
            # kernel2 = [[[[2/9]],[[1/9]]],
            #           [[[4/9]],[[2/9]]]]
            
            # output17,output18=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(2,2),kernel=kernel1)    #about binary predictions 50%
            # output19,output20=filtro_Cov2_size_kernel(image=output18,size_kernel=(3,3),stride_x_y=(1,1),kernel=kernel2)  #on binary predictions of % fixed    
            
            
            # image_copia_kernel=np.ones(((max_y2)+2,(max_x2)+2))
            
            # for filas in range(image_copia_kernel.shape[0]):
            #     for columnas in range(image_copia_kernel.shape[1]): 
                    
            #         if filas%2==0 and columnas%2==0:
            #             image_copia_kernel[filas][columnas]=1/9
                    
            #         if filas%2==0 and columnas%2==1:
            #             image_copia_kernel[filas][columnas]=2/9
                        
            #         if filas%2==1 and columnas%2==0:
            #             image_copia_kernel[filas][columnas]=2/9
                        
            #         if filas%2==1 and columnas%2==1:
            #             image_copia_kernel[filas][columnas]=4/9
            
            # kernel1 = [[[[1/8]],[[2/8]]],
            #           [[[2/8]],[[3/8]]]]
            
            # kernel2 = [[[[2/8]],[[1/8]]],
            #           [[[3/8]],[[2/8]]]]
            
            # kernel3 = [[[[2/8]],[[3/8]]],
            #           [[[1/8]],[[2/8]]]]
            
            # kernel4 = [[[[3/8]],[[2/8]]],
            #           [[[2/8]],[[1/8]]]]
            
            
            # kernel1 = [[[[1/5]],[[1/5]]],
            #           [[[1/5]],[[2/5]]]]
            
            # kernel2 = [[[[1/5]],[[1/5]]],
            #           [[[2/5]],[[1/5]]]]
            
            # kernel3 = [[[[1/5]],[[2/5]]],
            #           [[[1/5]],[[1/5]]]]
            
            # kernel4 = [[[[2/5]],[[1/5]]],
            #           [[[1/5]],[[1/5]]]]            
            
            
            # kernel1 = [[[[1/9]],[[2/9]]],
            #           [[[2/9]],[[4/9]]]]
            
            # kernel2 = [[[[2/9]],[[1/9]]],
            #           [[[4/9]],[[2/9]]]]
            
            # kernel3 = [[[[2/9]],[[4/9]]],
            #           [[[1/9]],[[2/9]]]]
            
            # kernel4 = [[[[4/9]],[[2/9]]],
            #           [[[2/9]],[[1/9]]]]
            
            
            kernel1 = [[[[2/10]],[[2/10]]],
                      [[[2/10]],[[4/10]]]]
            
            kernel2 = [[[[2/10]],[[2/10]]],
                      [[[4/10]],[[2/10]]]]
            
            kernel3 = [[[[2/10]],[[4/10]]],
                      [[[2/10]],[[2/10]]]]
            
            kernel4 = [[[[4/10]],[[2/10]]],
                      [[[2/10]],[[2/10]]]]
            
            
            output17_a,output18_a=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel1)    #sobre las predicciones binarias 50%
            output17_b,output18_b=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel2)    #sobre las predicciones binarias 50%
            output17_c,output18_c=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel3)    #sobre las predicciones binarias 50%
            output17_d,output18_d=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel4)    #sobre las predicciones binarias 50%

            
            output17=np.ones((output17_a.shape[0],output17_a.shape[1]))
            
            for filas in range(output17.shape[0]):
                for columnas in range(output17.shape[1]): 
                    
                    if filas%2==0 and columnas%2==0:
                        output17[filas][columnas]=output17_a[filas][columnas]
                    
                    if filas%2==0 and columnas%2==1:
                        output17[filas][columnas]=output17_b[filas][columnas]
                        
                    if filas%2==1 and columnas%2==0:
                        output17[filas][columnas]=output17_c[filas][columnas]
                        
                    if filas%2==1 and columnas%2==1:
                        output17[filas][columnas]=output17_d[filas][columnas]
            

            output17_reshape = output17.reshape(1, output17.shape[0], output17.shape[1], 1)

            output19 = modelll.predict(output17_reshape)
            
            output19 = np.squeeze(output19)
            
            output20=np.copy(output19)
            
            for i in range(output20.shape[0]):
                for j in range(output20.shape[1]):
                    
                    if output20[i][j]>=1/4:
                        output20[i][j]=1
                    else:
                        output20[i][j]=0 
            



            output21_a,output22_a=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel1)    #sobre las predicciones binarias 50%
            output21_b,output22_b=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel2)    #sobre las predicciones binarias 50%
            output21_c,output22_c=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel3)    #sobre las predicciones binarias 50%
            output21_d,output22_d=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel4)    #sobre las predicciones binarias 50%     
            
            
            output21=np.ones((output21_a.shape[0],output21_a.shape[1]))
            
            for filas in range(output21.shape[0]):
                for columnas in range(output21.shape[1]): 
                    
                    if filas%2==0 and columnas%2==0:
                        output21[filas][columnas]=output21_a[filas][columnas]
                    
                    if filas%2==0 and columnas%2==1:
                        output21[filas][columnas]=output21_b[filas][columnas]
                        
                    if filas%2==1 and columnas%2==0:
                        output21[filas][columnas]=output21_c[filas][columnas]
                        
                    if filas%2==1 and columnas%2==1:
                        output21[filas][columnas]=output21_d[filas][columnas]
            
            
            output21_reshape = output21.reshape(1, output21.shape[0], output21.shape[1], 1)
            output23 = modelll.predict(output21_reshape)
            output23 = np.squeeze(output23)
            
            output24=np.copy(output23)
            
            for i in range(output24.shape[0]):
                for j in range(output24.shape[1]):
                    
                    if output24[i][j]>=1/4:
                        output24[i][j]=1
                    else:
                        output24[i][j]=0 
            
            
    # Color map of cancer 
    
     #With this output I will calculate the map of the image, for this I will multiply the pred of each corner by 50 and I will locate it in the corresponding position. 
            #I use the 2x2,1 filter output, so I multiply by 50, it's as if each patch was divided in 4 sub_patches of 50x50.
    
            img7_=np.ones(((max_y)*100,(max_x)*100,3))
            x=50
            
            #I might have to change it because the validation images are ordered differently from the test images on the axes.            
            for i in tqdm(range(max_y2)):    #I cut the dimensions of 50 by 50  
                for j in range(max_x2):
            
                    if output6[i][j]<0.5:     # if in each 2x2 filtering of the averagepooling I get less than 0.5 it means that there are more healthy patches than with tumour, so I put the original image.
            
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,:]=img2_[i*x:(i+1)*x,j*x:(j+1)*x,:]
                    else: # If not, I put the prediction multiplied by 255 to indicate the zone.
            
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,0]=[255]
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,1]=[1-output6[i][j]*255]
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,2]=[0]
            
            
            etiquetas=[]      
            img8_=img7_.astype(np.uint8) 
            
            
            dst3 = cv2.addWeighteddst = cv2.addWeighted(img2_,0.7,img8_,0.3,0)     
    
    
            
            
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2')    
            matplotlib.pyplot.imshow(output)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 (salida etiquetas tumor and normal)')    
            matplotlib.pyplot.imshow(output2)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%')    
            matplotlib.pyplot.imshow(output3)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%'+' (salida etiquetas tumor and normal)')   
            matplotlib.pyplot.imshow(output4)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 con kernel')    
            matplotlib.pyplot.imshow(output13)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 (salida etiquetas tumor and normal) con kernel')   
            matplotlib.pyplot.imshow(output14)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'% con kernel')    
            matplotlib.pyplot.imshow(output15)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%'+' (salida etiquetas tumor and normal) con kernel')   
            matplotlib.pyplot.imshow(output16)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1')    
            matplotlib.pyplot.imshow(output6)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2')    
            matplotlib.pyplot.imshow(output7)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 (salida etiquetas tumor and normal)')   
            matplotlib.pyplot.imshow(output8)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - ' +str(limite_pred)+'%')  
            matplotlib.pyplot.imshow(output11)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - ' +str(limite_pred)+'%'+' round (al 50%)')   
            matplotlib.pyplot.imshow(output12)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 con kernel')    
            matplotlib.pyplot.imshow(output19)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 (salida etiquetas tumor and normal) con kernel')   
            matplotlib.pyplot.imshow(output20)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - '+str(limite_pred)+'% con kernel')    
            matplotlib.pyplot.imshow(output23)
            plt.show()
            matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - '+str(limite_pred)+'% (salida etiquetas tumor and normal) con kernel')   
            matplotlib.pyplot.imshow(output24)
            plt.show()
            
            matplotlib.pyplot.title('Mapa de calor Overlapping y averagepooling filtrado 2x2,1')      
            matplotlib.pyplot.imshow(dst3)
            plt.show()
    
            
            
            
            labels_real_2_1=aa.flatten()
            label_preds_overlapping_1_1 = output2.flatten()
            label_preds_overlapping_2_1 = output4.flatten()
            label_preds_overlapping_3_1 = output8.flatten()
            label_preds_overlapping_4_1 = output12.flatten()
            label_preds_overlapping_5_1 = output14.flatten()
            label_preds_overlapping_6_1 = output16.flatten()
            label_preds_overlapping_7_1 = output20.flatten()
            label_preds_overlapping_8_1 = output24.flatten()
            
            
            
            label_preds_overlapping_1_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_1_1, labels_real_2_1)
            label_preds_overlapping_2_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_2_1, labels_real_2_1)
            label_preds_overlapping_3_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_3_1, labels_real_2_1)
            label_preds_overlapping_4_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_4_1, labels_real_2_1)
            label_preds_overlapping_5_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_5_1, labels_real_2_1)
            label_preds_overlapping_6_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_6_1, labels_real_2_1)
            label_preds_overlapping_7_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_7_1, labels_real_2_1)
            label_preds_overlapping_8_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_8_1, labels_real_2_1)







            labels_real_2=labels_real_2_1_sin_blancos
            labels_real_2_append.extend(labels_real_2)
            label_preds_overlapping = label_preds_overlapping_1_1_1
            label_preds_overlapping_2 = label_preds_overlapping_2_1_1
            label_preds_overlapping_3 = label_preds_overlapping_3_1_1
            label_preds_overlapping_4 = label_preds_overlapping_4_1_1
            label_preds_overlapping_5 = label_preds_overlapping_5_1_1
            label_preds_overlapping_6 = label_preds_overlapping_6_1_1
            label_preds_overlapping_7 = label_preds_overlapping_7_1_1
            label_preds_overlapping_8 = label_preds_overlapping_8_1_1
            
            
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 3x3,2 = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping)
            print(classification_report(labels_real_2, label_preds_overlapping, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping)))
            print('\n')
            print(cm)   
            
    
            label_preds_overlapping_append.extend(label_preds_overlapping) 
                    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_2)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 3x3,2 '+str(limite_pred)+'%'+' = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_2)
            print(classification_report(labels_real_2, label_preds_overlapping_2, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_2)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_2)))
            print('\n')
            print(cm)   
            print('\n \n')
            
            label_preds_overlapping_2_append.extend(label_preds_overlapping_2) 
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_5)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 3x3,2 con kernel= '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_5)
            print(classification_report(labels_real_2, label_preds_overlapping_5, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_5)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_5)))
            print('\n')
            print(cm)               
    
            label_preds_overlapping_5_append.extend(label_preds_overlapping_5) 
                    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_6)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 3x3,2 '+str(limite_pred)+'% con kernel'+' = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_6)
            print(classification_report(labels_real_2, label_preds_overlapping_6, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_6)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_6)))
            print('\n')
            print(cm)   
            print('\n \n')
            
            label_preds_overlapping_6_append.extend(label_preds_overlapping_6) 
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_3)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 2x2,1 + 2x2,2 = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_3)
            print(classification_report(labels_real_2, label_preds_overlapping_3, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_3)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_3)))
            print('\n')
            print(cm)               
    
            label_preds_overlapping_3_append.extend(label_preds_overlapping_3) 
                    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_4)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 2x2,1 + 2x2,2 '+str(limite_pred)+'%'+' = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_4)
            print(classification_report(labels_real_2, label_preds_overlapping_4, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_4)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_4)))
            print('\n')
            print(cm)   
            print('\n \n')
            
            label_preds_overlapping_4_append.extend(label_preds_overlapping_4) 
              
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_7)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 2x2,1 + 2x2,2  con kernel= '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_7)
            print(classification_report(labels_real_2, label_preds_overlapping_7, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_7)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_7)))
            print('\n')
            print(cm)               
    
            label_preds_overlapping_7_append.extend(label_preds_overlapping_7)            
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_8)
            print('Accuracy salida binaria imagen 100x100 con overlapping + averagepooling 2x2,1 + 2x2,2 '+str(limite_pred)+'% con kernel'+' = '+str(acc))
            print('CM CNN imagen '+nombre_imagenes_test[w]+'_idx5.png')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_8)
            print(classification_report(labels_real_2, label_preds_overlapping_8, target_names=classes2))
            print('\n')
            print('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_8)))
            print('\n')
            print('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_8)))
            print('\n')
            print(cm)   
            print('\n \n')
            
            label_preds_overlapping_8_append.extend(label_preds_overlapping_8) 
            
    ################################## Data storage in Txt 

            newfile = open(path_output_test+name_model+version+".txt", "a+")
    
            
            newfile.write('Imagen '+nombre_imagenes_test[w]+'_idx5.png'+'\n \n \n')
            
            acc=accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)
            newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)))
            newfile.write('\n')
            cm = confusion_matrix(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos)
            
            newfile.write(classification_report(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_sin_blancos, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
            
            acc=accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)
            newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping con un porcentaje de predicción de tumor mayor al ' +str(limite_pred)+'% = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)))
            newfile.write('\n')
            cm = confusion_matrix(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos)
            newfile.write('\n')
            newfile.write(classification_report(etiquetas_reales_sin_blancos, etiquetas_pred_binarias_porcentaje_fijado_sin_blancos, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
    
            acc=accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)))
            newfile.write('\n')
            cm = confusion_matrix(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos)
            newfile.write(classification_report(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_sin_blancos, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
             
             
            acc=accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping con un porcentaje de predicción de tumor mayor al ' +str(limite_pred)+'% = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)))
            newfile.write('\n')
            cm = confusion_matrix(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos)
            newfile.write(classification_report(etiquetas_reales2_sin_blancos, etiquetas_pred_binarias2_porcentaje_fijado_sin_blancos, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
     
            acc=accuracy_score(labels_real_2, label_preds_overlapping)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_2)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) > '+str(limite_pred)+'%'+' = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_2)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_2)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_2)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_2, target_names=classes2))
            newfile.write(str(cm)+'\n')
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_5)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) con kernel = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_5)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_5)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_5)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_5, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_6)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) > '+str(limite_pred)+'% con kernel'+' = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_6)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_6)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_6)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_6, target_names=classes2))
            newfile.write(str(cm)+'\n')
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_3)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_3)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_3)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_3)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_3, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_4)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'%'+' = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_4)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_4)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_4)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_4, target_names=classes2))
            newfile.write(str(cm)+'\n')
            
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_7)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) con kernel = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_7)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_7)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_7)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_7, target_names=classes2))
            newfile.write(str(cm)+'\n')
            newfile.write('\n')
    
            
            acc=accuracy_score(labels_real_2, label_preds_overlapping_8)
            newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'%  con kernel'+' = '+str(acc))
            newfile.write('\n')
            newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2, label_preds_overlapping_8)))
            newfile.write('\n')
            newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2, label_preds_overlapping_8)))
            newfile.write('\n')
            cm = confusion_matrix(labels_real_2, label_preds_overlapping_8)
            newfile.write(classification_report(labels_real_2, label_preds_overlapping_8, target_names=classes2))
            newfile.write(str(cm)+'\n')
            
            
            
            newfile.write('------------------------------------------------------------------------------------'+'\n')

            
            newfile.write('\n \n \n \n ')
            
            
            newfile.close()
            
            
            z=z+max_x*max_y
            z1=z1+max_x2*max_y2
    
    
    
    etiquetas_reales_sin_blanco_append=np.array(etiquetas_reales_sin_blanco_append)
    etiquetas_pred_binarias_append=np.array(etiquetas_pred_binarias_append)
    etiquetas_pred_binarias_porcentaje_fijado_append=np.array(etiquetas_pred_binarias_porcentaje_fijado_append)
    etiquetas_reales2_append=np.array(etiquetas_reales2_append)
    etiquetas_pred_binarias2_append=np.array(etiquetas_pred_binarias2_append)
    etiquetas_pred_binarias_porcentaje_fijado2_append=np.array(etiquetas_pred_binarias_porcentaje_fijado2_append)
    label_preds_overlapping_append=np.array(label_preds_overlapping_append)
    label_preds_overlapping_2_append=np.array(label_preds_overlapping_2_append)
    label_preds_overlapping_3_append=np.array(label_preds_overlapping_3_append)
    label_preds_overlapping_4_append=np.array(label_preds_overlapping_4_append)
    label_preds_overlapping_5_append=np.array(label_preds_overlapping_5_append)
    label_preds_overlapping_6_append=np.array(label_preds_overlapping_6_append)
    label_preds_overlapping_7_append=np.array(label_preds_overlapping_7_append)
    label_preds_overlapping_8_append=np.array(label_preds_overlapping_8_append)

    
    
    
    newfile = open(path_output_test+name_model+version+".txt", "a+")
    
    
    newfile.write('RESUMEN DEL TESTEO TOTAL DE CADA SPLIT'+'\n \n \n')
    
    

    
    acc=accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append)
    newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append)))
    newfile.write('\n')
    cm = confusion_matrix(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append)
    newfile.write(classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normalfloat,tumorfloat,accuracy_float)
    
    BACC_vector= np.append(BACC_float, balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append))
    MCC_vector= np.append(MCC_float,matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append))


    
    acc=accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append)
    newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping con un porcentaje de predicción de tumor mayor al ' +str(limite_pred)+'% = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append)))
    newfile.write('\n')
    cm = confusion_matrix(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append)
    newfile.write(classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append))
    
    
    acc=accuracy_score(etiquetas_reales2_append, etiquetas_pred_binarias2_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_append, etiquetas_pred_binarias2_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_append, etiquetas_pred_binarias2_append)))
    newfile.write('\n')
    cm = confusion_matrix(etiquetas_reales2_append, etiquetas_pred_binarias2_append)
    newfile.write(classification_report(etiquetas_reales2_append, etiquetas_pred_binarias2_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
     
     
    acc=accuracy_score(etiquetas_reales2_append, etiquetas_pred_binarias_porcentaje_fijado2_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping con un porcentaje de predicción de tumor mayor al' +str(limite_pred)+'% = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_append, etiquetas_pred_binarias_porcentaje_fijado2_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_append, etiquetas_pred_binarias_porcentaje_fijado2_append)))
    newfile.write('\n')
    cm = confusion_matrix(etiquetas_reales2_append, etiquetas_pred_binarias_porcentaje_fijado2_append)
    newfile.write(classification_report(etiquetas_reales2_append, etiquetas_pred_binarias_porcentaje_fijado2_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_append))
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_2_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) >'+str(limite_pred)+'%'+' = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_2_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_2_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_2_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_2_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_2_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_2_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_2_append))
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_5_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) con kernel = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_5_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_5_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_5_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_5_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_5_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_5_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_5_append))
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_6_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) >'+str(limite_pred)+'%'+' con kernel = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_6_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_6_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_6_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_6_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_6_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_6_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_6_append))
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_3_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_3_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_3_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_3_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_3_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_3_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_3_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_3_append))
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_4_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'%'+' = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_4_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_4_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_4_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_4_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_4_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_4_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_4_append))
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_7_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) con kernel= '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_7_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_7_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_7_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_7_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_7_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_7_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_7_append))
    
    
    
    acc=accuracy_score(labels_real_2_append, label_preds_overlapping_8_append)
    newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'% con kernel'+' = '+str(acc))
    newfile.write('\n')
    newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_8_append)))
    newfile.write('\n')
    newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_append, label_preds_overlapping_8_append)))
    newfile.write('\n')
    cm = confusion_matrix(labels_real_2_append, label_preds_overlapping_8_append)
    newfile.write(classification_report(labels_real_2_append, label_preds_overlapping_8_append, target_names=classes2))
    newfile.write(str(cm)+'\n')
    newfile.write('\n')
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_8_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_8_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_8_append))
    
    
    
    newfile.write('------------------------------------------------------------------------------------'+'\n')
    newfile.write('------------------------------------------------------------------------------------'+'\n')
    newfile.write('------------------------------------------------------------------------------------'+'\n')
    
    tablas_tumor['split_'+str(split)+'_tumor'] = tumor_float_vector
    tablas_normal['split_'+str(split)+'_normal'] = normal_float_vector
    tablas_accuracy['split_'+str(split)+'_accuracy'] = accuracy_vector
    tablas_BACC['split_'+str(split)+'_BACC'] = BACC_vector
    tablas_MCC['split_'+str(split)+'_MCC'] = MCC_vector


    
    newfile.write('\n \n \n \n ')
    
    
    newfile.close()
    
    etiquetas_reales_total.extend(etiquetas_reales_sin_blanco_append)
    etiquetas_pred_binarias_total.extend(etiquetas_pred_binarias_append)
    etiquetas_pred_binarias_porcentaje_fijado_total.extend(etiquetas_pred_binarias_porcentaje_fijado_append)
    etiquetas_reales2_total.extend(etiquetas_reales2_append)
    etiquetas_pred_binarias_porcentaje_fijado2_total.extend(etiquetas_pred_binarias_porcentaje_fijado2_append)
    etiquetas_pred_binarias2_total.extend(etiquetas_pred_binarias2_append)
    labels_real_2_total.extend(labels_real_2_append)

    label_preds_overlapping_total.extend(label_preds_overlapping_append)
    label_preds_overlapping_2_total.extend(label_preds_overlapping_2_append)
    label_preds_overlapping_3_total.extend(label_preds_overlapping_3_append)
    label_preds_overlapping_4_total.extend(label_preds_overlapping_4_append)
    label_preds_overlapping_5_total.extend(label_preds_overlapping_5_append)
    label_preds_overlapping_6_total.extend(label_preds_overlapping_6_append)
    label_preds_overlapping_7_total.extend(label_preds_overlapping_7_append)
    label_preds_overlapping_8_total.extend(label_preds_overlapping_8_append)

  


tumor_tabla=np.stack((tablas_tumor['split_0_tumor'],tablas_tumor['split_1_tumor'],tablas_tumor['split_2_tumor'],tablas_tumor['split_3_tumor'],tablas_tumor['split_4_tumor']),axis=-1)
normal_tabla=np.stack((tablas_normal['split_0_normal'],tablas_normal['split_1_normal'],tablas_normal['split_2_normal'],tablas_normal['split_3_normal'],tablas_normal['split_4_normal']),axis=-1)
accuracy_tabla=np.stack((tablas_accuracy['split_0_accuracy'],tablas_accuracy['split_1_accuracy'],tablas_accuracy['split_2_accuracy'],tablas_accuracy['split_3_accuracy'],tablas_accuracy['split_4_accuracy']),axis=-1)
BACC_tabla=np.stack((tablas_BACC['split_0_BACC'],tablas_BACC['split_2_BACC'],tablas_BACC['split_2_BACC'],tablas_BACC['split_3_BACC'],tablas_BACC['split_4_BACC']),axis=-1)
MCC_tabla=np.stack((tablas_MCC['split_0_MCC'],tablas_MCC['split_1_MCC'],tablas_MCC['split_2_MCC'],tablas_MCC['split_3_MCC'],tablas_MCC['split_4_MCC']),axis=-1)


performance_accuracy=np.transpose(accuracy_tabla)
performance_BACC=np.transpose(BACC_tabla)
performance_MCC=np.transpose(MCC_tabla)


Tabla_normal_flatten=normal_tabla.flatten()
Tabla_tumor_flatten=tumor_tabla.flatten()

  
newfile = open(path_output_test+"output_testeo_images_RESUMEN_"+name_model+version+".txt", "a+")


newfile.write('RESUMEN DEL TESTEO TOTAL'+'\n \n \n')

acc=accuracy_score(etiquetas_reales_total, etiquetas_pred_binarias_total)
newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_total, etiquetas_pred_binarias_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_total, etiquetas_pred_binarias_total)))
newfile.write('\n')
cm = confusion_matrix(etiquetas_reales_total, etiquetas_pred_binarias_total)
newfile.write(classification_report(etiquetas_reales_total, etiquetas_pred_binarias_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total)
newfile.write('Accuracy salida binaria imagen 100x100 sin overlapping con un porcentaje de predicción de tumor mayor al ' +str(limite_pred)+'% = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total)))
newfile.write('\n')
cm = confusion_matrix(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total)
newfile.write(classification_report(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(etiquetas_reales2_total, etiquetas_pred_binarias2_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_total, etiquetas_pred_binarias2_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_total, etiquetas_pred_binarias2_total)))
newfile.write('\n')
cm = confusion_matrix(etiquetas_reales2_total, etiquetas_pred_binarias2_total)
newfile.write(classification_report(etiquetas_reales2_total, etiquetas_pred_binarias2_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')
 
 
acc=accuracy_score(etiquetas_reales2_total, etiquetas_pred_binarias_porcentaje_fijado2_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping con un porcentaje de predicción de tumor mayor al' +str(limite_pred)+'% = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(etiquetas_reales2_total, etiquetas_pred_binarias_porcentaje_fijado2_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(etiquetas_reales2_total, etiquetas_pred_binarias_porcentaje_fijado2_total)))
newfile.write('\n')
cm = confusion_matrix(etiquetas_reales2_total, etiquetas_pred_binarias_porcentaje_fijado2_total)
newfile.write(classification_report(etiquetas_reales2_total, etiquetas_pred_binarias_porcentaje_fijado2_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_2_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) >'+str(limite_pred)+'%'+' = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_2_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_2_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_2_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_2_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_5_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) con kernel = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_5_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_5_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_5_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_5_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_6_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 3x3,2) >'+str(limite_pred)+'%'+' con kernel = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_6_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_6_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_6_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_6_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')



acc=accuracy_score(labels_real_2_total, label_preds_overlapping_3_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_3_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_3_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_3_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_3_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_4_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'%'+' = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_4_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_4_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_4_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_4_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')


acc=accuracy_score(labels_real_2_total, label_preds_overlapping_7_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) con kernel= '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_7_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_7_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_7_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_7_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')
    
    
acc=accuracy_score(labels_real_2_total, label_preds_overlapping_8_total)
newfile.write('Accuracy salida binaria imagen 100x100 con overlapping (tras averagepooling 2x2,1 + 2x2,2) > '+str(limite_pred)+'% con kernel'+' = '+str(acc))
newfile.write('\n')
newfile.write('BACC salida binaria imagen 100x100 sin overlapping = ' +str(balanced_accuracy_score(labels_real_2_total, label_preds_overlapping_8_total)))
newfile.write('\n')
newfile.write('MCC salida binaria imagen 100x100 sin overlapping = ' +str(matthews_corrcoef(labels_real_2_total, label_preds_overlapping_8_total)))
newfile.write('\n')
cm = confusion_matrix(labels_real_2_total, label_preds_overlapping_8_total)
newfile.write(classification_report(labels_real_2_total, label_preds_overlapping_8_total, target_names=classes2))
newfile.write(str(cm)+'\n')
newfile.write('\n')
    

newfile.write('------------------------------------------------------------------------------------')


newfile.write('\n \n \n \n ')


newfile.close()

normal_tabla_precision=[]
tumor_tabla_precision=[]


for i in range(0,len(normal_tabla),3):
    
    normal_tabla_precision.append(normal_tabla[i])
    tumor_tabla_precision.append(tumor_tabla[i])

    
normal_tabla_precision_array=np.array(normal_tabla_precision)
tumor_tabla_precision_array=np.array(tumor_tabla_precision)

normal_tabla_precision_array_flatten=normal_tabla_precision_array.flatten()
tumor_tabla_precision_array_flatten=tumor_tabla_precision_array.flatten()


normal_tabla_recall=[]
tumor_tabla_recall=[]




for i in range(1,len(normal_tabla),3):
    
    normal_tabla_recall.append(normal_tabla[i])
    tumor_tabla_recall.append(tumor_tabla[i])

    
normal_tabla_recall_array=np.array(normal_tabla_recall)
tumor_tabla_recall_array=np.array(tumor_tabla_recall)


normal_tabla_recall_array_flatten=normal_tabla_recall_array.flatten()
tumor_tabla_recall_array_flatten=tumor_tabla_recall_array.flatten()




normal_tabla_f1_score=[]
tumor_tabla_f1_score=[]


for i in range(2,len(normal_tabla),3):
    
    normal_tabla_f1_score.append(normal_tabla[i])
    tumor_tabla_f1_score.append(tumor_tabla[i])

    
normal_tabla_f1_score_array=np.array(normal_tabla_f1_score)
tumor_tabla_f1_score_array=np.array(tumor_tabla_f1_score)


normal_tabla_f1_score_array_flatten=normal_tabla_f1_score_array.flatten()
tumor_tabla_f1_score_array_flatten=tumor_tabla_f1_score_array.flatten()


performance_markers_normal=np.stack((normal_tabla_precision_array_flatten, normal_tabla_recall_array_flatten,normal_tabla_f1_score_array_flatten),-1)
performance_markers_tumor=np.stack((tumor_tabla_precision_array_flatten, tumor_tabla_recall_array_flatten,tumor_tabla_f1_score_array_flatten),-1)

performance_markers=np.concatenate((performance_markers_normal,performance_markers_tumor))

MCC_media=np.mean(performance_MCC,0)
BACC_media=np.mean(performance_BACC,0)

# # -*- coding: utf-8 -*-
# """
# Created on Tue May 24 17:07:30 2022

# @author: sergio
# """

# img3_=np.ones(((max_y)*100,(max_x)*100,3))



# for i in tqdm(range(max_y)):
#     for j in range(max_x):

#         if etiquetas_pred_porcentajes_matriz[i][j]<0.5:

#             img3_[i*x:(i+1)*x,j*x:(j+1)*x,:]=img2_[i*x:(i+1)*x,j*x:(j+1)*x,:]
            
#         else:

#             img3_[i*x:(i+1)*x,j*x:(j+1)*x,0]=255
#             img3_[i*x:(i+1)*x,j*x:(j+1)*x,1]=[int(1-((etiquetas_pred_porcentajes_matriz[i][j]-0.5)*2)*255)]
#             img3_[i*x:(i+1)*x,j*x:(j+1)*x,2]=28


# etiquetas=[]      
# img3_=img3_.astype(np.uint8) 


# dst = cv2.addWeighteddst = cv2.addWeighted(img3_,0.6,img2_,0.3,3)     

    
# matplotlib.pyplot.title('Imagen Original')    
# matplotlib.pyplot.imshow(img3_,cmap='YlOrRd')
# plt.show()

# matplotlib.pyplot.title('Imagen Original')    
# matplotlib.pyplot.imshow(dst,cmap='YlOrRd')
# plt.show()

from sklearn.metrics import roc_curve
import sklearn.metrics as metrics


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

fper1, tper1, thresholds = roc_curve(etiquetas_reales_total, etiquetas_pred_binarias_total)
roc_auc1 = metrics.auc(fper1, tper1)

fper2, tper2, thresholds = roc_curve(etiquetas_reales_total, etiquetas_pred_binarias_porcentaje_fijado_total)
roc_auc2 = metrics.auc(fper2, tper2)

fper3, tper3, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_total)
roc_auc3 = metrics.auc(fper3, tper3)

fper4, tper4, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_2_total)
roc_auc4 = metrics.auc(fper4, tper4)

fper5, tper5, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_5_total)
roc_auc5 = metrics.auc(fper5, tper5)

fper6, tper6, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_6_total)
roc_auc6 = metrics.auc(fper6, tper6)

fper7, tper7, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_3_total)
roc_auc7 = metrics.auc(fper7, tper7)

fper8, tper8, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_4_total)
roc_auc8 = metrics.auc(fper8, tper8)

fper9, tper9, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_7_total)
roc_auc9 = metrics.auc(fper9, tper9)

fper10, tper10, thresholds = roc_curve(labels_real_2_total, label_preds_overlapping_8_total)
roc_auc10 = metrics.auc(fper10, tper10)


plt.plot(fper1, tper1,'r', label='ROC_1 AUC = %0.3f' % roc_auc1)
plt.plot(fper2, tper2,'b', label='ROC_2 AUC = %0.3f' % roc_auc2)
plt.plot(fper3, tper3,'y', label='ROC_3 AUC = %0.3f' % roc_auc3)
plt.plot(fper4, tper4,'m', label='ROC_4 AUC = %0.3f' % roc_auc4)
plt.plot(fper5, tper5,'g', label='ROC_5 AUC = %0.3f' % roc_auc5 )
plt.plot(fper6, tper6,'gold', label='ROC_6 AUC = %0.3f' % roc_auc6)
plt.plot(fper7, tper7,'k', label='ROC_7 AUC = %0.3f' % roc_auc7)
plt.plot(fper8, tper8,'c', label='ROC_8 AUC = %0.3f' % roc_auc8)
plt.plot(fper9, tper9,'burlywood',label='ROC_9 AUC = %0.3f' % roc_auc9)
plt.plot(fper10, tper10,'brown', label='ROC_10 AUC = %0.3f' % roc_auc10)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()


Acurracy_final=[]

for i in range(np.size(performance_accuracy,1)):
               
     Acurracy_final.append(np.mean(performance_accuracy[0:5,i],dtype=np.float32))
     
BACC_final=[]

for i in range(np.size(performance_BACC,1)):
               
     BACC_final.append(np.mean(performance_BACC[0:5,i],dtype=np.float32))

MCC_final=[]

for i in range(np.size(performance_MCC,1)):
               
     MCC_final.append(np.mean(performance_MCC[0:5,i],dtype=np.float32))


 
