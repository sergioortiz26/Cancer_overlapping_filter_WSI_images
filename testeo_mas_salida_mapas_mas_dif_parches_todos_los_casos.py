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
from elimina_parches_aislados import elimina_parches_aislados

# ---------------------------------------------------- INICIALIZACION DE VARIABLES


#carpeta_modelo_concreto='DenseNet121_ADAM_trainable_20%'
carpeta_modelo_concreto='DenseNet121_SGD_trainable_70%'


#path_model='C:/Users/Sergio/Documents/CANCER/script/model_ResNet50_50_epochs_256_batchsize_Adam_trainable_20%_optimiz_100x100_inputs_labels_split_path_0-100x100_rgb_15_porc_sin_14_test.h5'
#path_file_model='C:/Users/Sergio/Documents/CANCER/modelos/'

path_file_model='C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/modelos3/'
#path_model='C:/Users/Sergio/Documents/CANCER/script/model_VGG16_50_epochs_256_batchsize_ADAM_optimiz_100x100_inputs_labels_split_path_0-100x100_rgb_15_porc_sin_14_test.h5'

path_imagenes='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_imgs_idx5/CINJ_imgs_idx5/'
#path_imagenes='C:/Users/Sergio/Documents/CANCER/HUP_images/'


path_inputs_patches='C:/Users/Sergio/Documents/CANCER/testeo_con_imagenes_CINJ/inputs_test/'
#path_inputs_patches='C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/inputs_train_test_validation3/'

path_output_test='C:/Users/Sergio/Desktop/'

path_mascaras='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_masks_HG/CINJ_masks_HG/'
#path_mascaras='C:/Users/Sergio/Documents/CANCER/HUP_masks/'


# path_imagenes='C:/Users/Sergio/Documents/CANCER/HUP_images/'
# path_inputs_patches='C:/Users/Sergio/Documents/CANCER/balanceo/inputs_train_test_validation/'
# path_mascaras='C:/Users/Sergio/Documents/CANCER/HUP_masks/'


version='_version_70%_V1'


#nombre_imagenes_test=['12880','12886','8863','8864','8865','8867']


classes = ['tumor', 'normal']
classes2=['normal','tumor']


limite_pred=70
splits=5


# ---------------------------------------------------- LECTURA DE LOS PARCHES DE LAS IMAGENES DE TEST SIN OVERLAPPING


etiquetas_reales_total=[]
etiquetas_pred_binarias_total=[]
etiquetas_pred_binarias_porcentaje_fijado_total=[]
etiquetas_reales2_total=[]
etiquetas_pred_binarias_porcentaje_fijado2_total=[]
etiquetas_pred_binarias2_total=[]
labels_real_2_total=[]


etiquetas_pred_binarias_sin_blancos_total=[]


label_preds_overlapping_total=[]
label_preds_overlapping_2_total=[]
label_preds_overlapping_3_total=[]
label_preds_overlapping_4_total=[]
label_preds_overlapping_5_total=[]
label_preds_overlapping_6_total=[]
label_preds_overlapping_7_total=[]
label_preds_overlapping_8_total=[]
    
label_preds_overlapping_total_parches_aislados=[]
label_preds_overlapping_2_total_parches_aislados=[]
label_preds_overlapping_3_total_parches_aislados=[]
label_preds_overlapping_4_total_parches_aislados=[]
label_preds_overlapping_5_total_parches_aislados=[]
label_preds_overlapping_6_total_parches_aislados=[]
label_preds_overlapping_7_total_parches_aislados=[]
label_preds_overlapping_8_total_parches_aislados=[]
    
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


    Data_sin_overlapping=data_reader('DATA split '+str(split)+' imagenes test sin overlapping')   #me creo un objeto data

    path_patches_sin_overlapping=path_inputs_patches+'inputs_labels_split_path_test_sin_overlapping_split_'+str(split)+'_100x100.npy'
    Data_sin_overlapping.save_data(path_patches_sin_overlapping,'test',classes)    # llamo al metodo save data del objeto, para guardar los datos (indicando que es lo que quiero leer, en este caso los datos de 'test')
    
    x_test_sin_overlapping = Data_sin_overlapping.data['test']['x_test']
    y_test_sin_overlapping = Data_sin_overlapping.data['test']['y_test']
    name_image_test=Data_sin_overlapping.data['test']['name_imges_test']

    nombre_imagenes_test=name_image_test    #ver el formato de salida de esta variable y compararlo con el de otro script de tsteo

    # ---------------------------------------------------- LECTURA DE LOS PARCHES DE LAS IMAGENES DE TEST CON OVERLAPPING
    
    Data_con_overlapping=data_reader('data1 14 imagenes sin overlapping')
    
    path_patches_con_overlapping=path_inputs_patches+'inputs_labels_split_path_test_con_overlapping_split_'+str(split)+'_100x100.npy'
    Data_con_overlapping.save_data(path_patches_con_overlapping,'test',classes)
    
    x_test_con_overlapping = Data_con_overlapping.data['test']['x_test']
    y_test_con_overlapping = Data_con_overlapping.data['test']['y_test']
    name_image_test=Data_sin_overlapping.data['test']['name_imges_test']
    
    
    #Inicializamos las variables que vamos a utilizar
    
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
    
    etiquetas_pred_binarias_sin_blancos_append=[]

    label_preds_overlapping_append_parches_aislados=[]
    label_preds_overlapping_2_append_parches_aislados=[]
    label_preds_overlapping_3_append_parches_aislados=[]
    label_preds_overlapping_4_append_parches_aislados=[]
    label_preds_overlapping_5_append_parches_aislados=[]
    label_preds_overlapping_6_append_parches_aislados=[]
    label_preds_overlapping_7_append_parches_aislados=[]
    label_preds_overlapping_8_append_parches_aislados=[]

    
    w=0
    z=0             # z es el contador de los inicios de las imagenes sin overlapping
    z1=0            # z1 es el contador de los inicios de las imagenes con overlapping
    
    
    
    
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
    
            # Selecciono de todos los datos de las imagenes de test, solo los de la imagen en question
            
            y_test_sin_overlapping_imagen_seleccionada,x_test_sin_overlapping_imagen_seleccionada,dimensions_image_sin_overlapping=data_of_imagen_in_question(
                                                                                                                                        img_,step_patch=100,position_data=z,y_test=y_test_sin_overlapping,x_test=x_test_sin_overlapping)
            
                            
            max_x=dimensions_image_sin_overlapping['max_x']
            max_y=dimensions_image_sin_overlapping['max_y']
    
            # Para esa imagen predigo las etiquetas de cada uno de sus parches
            
            [etiquetas_reales,etiquetas_pred_binarias,etiquetas_pred_porcentajes,etiquetas_pred_porcentaje_fijado,etiquetas_pred_binarias_porcentaje_fijado]=testeo_patches(
                                                                                                                                        x_test_sin_overlapping_imagen_seleccionada,y_test_sin_overlapping_imagen_seleccionada,path_model,limite_pred)
            
            
            etiquetas_pred_binarias_sin_blancos,etiquetas_reales_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, etiquetas_pred_binarias, etiquetas_reales)
            etiquetas_pred_binarias_porcentaje_fijado_sin_blancos,etiquetas_reales_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, etiquetas_pred_binarias_porcentaje_fijado, etiquetas_reales)
            
            etiquetas_reales_sin_blanco_append.extend(etiquetas_reales_sin_blancos)
            
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
            
            
            #MUY IMPORTANTE- CAMBIAR CUANDO SE UTILICE HUP
            img_mask=cv2.imread(path_mascaras+nombre_imagenes_test[w]+'.png')
            #img_mask=cv2.imread(path_mascaras+nombre_imagenes_test[w]+'_annotation_mask.png')

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
    
                
            # matplotlib.pyplot.title('Imagen Original')    
            # matplotlib.pyplot.imshow(img_)
            # plt.show()
            # matplotlib.pyplot.title('Mascara')    
            # matplotlib.pyplot.imshow(img_mask)
            # plt.show() 
            # matplotlib.pyplot.title('Mascara parches')    
            # matplotlib.pyplot.imshow(aa)
            # plt.show()
            # matplotlib.pyplot.title('Decision 50%')    
            # matplotlib.pyplot.imshow(cc)
            # plt.show()
            # matplotlib.pyplot.title('% pred de CNN')    
            # matplotlib.pyplot.imshow(gg)
            # plt.colorbar()
            # plt.show()
            # matplotlib.pyplot.title('Decision > 70%')    
            # matplotlib.pyplot.imshow(ii)
            # plt.show()        
            # matplotlib.pyplot.title('Dif entre real y 50%')    
            # matplotlib.pyplot.imshow(ee)
            # plt.show()
            # matplotlib.pyplot.title('Mapa de calor de la pred del 70%')    
            # matplotlib.pyplot.imshow(dst)
            # plt.show()
            # matplotlib.pyplot.imshow(dst2)
            # plt.show()
    
    
    
    ###### -------------------------------------------------------------------------------- IMAGENES CON OVERLAPPING ---------------------------------------------------------------------------------------------------- 
    
    
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
            
            # matplotlib.pyplot.title('Overlapping Mascara parches')    
            # matplotlib.pyplot.imshow(ll)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping Decision 50%')    
            # matplotlib.pyplot.imshow(mm)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping % pred de CNN')    
            # matplotlib.pyplot.imshow(nn)
            # plt.colorbar()
            # plt.show()
            
    
    
    ####################### FILTROS 3x3 con stride 2 ##########################
    
    #Filtros sobre las predicciones binarias 50%
    
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
                    
                    if output2[i][j]>=4.5/9:
                        output2[i][j]=1
                    else:
                        output2[i][j]=0 
    
    
    #Filtros sobre las predicciones binarias del % fijado
    
    
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
                    
                    if output4[i][j]>=4.5/9:   # fixed 4/9 because if 4 to 9 patch are tumor consider all with tumor (4 patches conform each corner of patch originalbinarizamos )
                        output4[i][j]=1
                    else:
                        output4[i][j]=0
                        
            
    ############ APPLY KERNEL ############
    
            kernel = [[[[1/16]],[[2/16]],[[1/16]]],
                      [[[2/16]],[[4/16]],[[2/16]]],
                      [[[1/16]],[[2/16]],[[1/16]]]]
            
            output13,output14=filtro_Cov2_size_kernel(image=image,size_kernel=(3,3),stride_x_y=(2,2),kernel=kernel)    #sobre las predicciones binarias 50%
            output15,output16=filtro_Cov2_size_kernel(image=image1,size_kernel=(3,3),stride_x_y=(2,2),kernel=kernel)   #sobre las predicciones binarias del % fijado
            
                        
            
            for i in range(output13.shape[0]):
                for j in range(output13.shape[1]):
                    
                    if output13[i][j]>=0.51:
                        output14[i][j]=1
                    else:
                        output14[i][j]=0 
                
                            
                                    
            for i in range(output15.shape[0]):
                for j in range(output15.shape[1]):
                    
                    if output15[i][j]>=0.51:
                        output16[i][j]=1
                    else:
                        output16[i][j]=0 
                
    
            
    ####################### FILTROS 2X2 con stride 1 + FILTRO 2x2 con stride 2 ##########################
            
    # (50%)
    # make the filter 2x2 with stride 1 to calculate the average of each corner of the image
    
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 1)])
    
            # generate pooled output
            output5 = modelll.predict(image) # Use 'image' due to in this variable it is store the matrix prediction 'overlapping' with 50% (with padding)
            output6 = np.squeeze(output5)
            
            for i in range(output6.shape[0]):
                for j in range(output6.shape[1]):
                    
                    if output6[i][j]>=0.51:
                        output6[i][j]=1
                    else:
                        output6[i][j]=0 
    
    
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
                    
                    if output8[i][j]>1/4:
                        output8[i][j]=1
                    else:
                        output8[i][j]=0
                        
                        
            
                        
            
    # (% fixed)
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 1)])
    
            # generate pooled output
            output9 = modelll.predict(image1)  # Use 'image1' due to in this variable it is store the matrix prediction 'overlapping' with % fixed (with padding)
            output10 = np.squeeze(output9)
            
            for i in range(output10.shape[0]):
                for j in range(output10.shape[1]):
                    
                    if output10[i][j]>=0.51:
                        output10[i][j]=1
                    else:
                        output10[i][j]=0 
    
    
            image2 = output10.reshape(1, output10.shape[0], output10.shape[1], 1)
    
    
            modelll = Sequential(
                [AveragePooling2D(pool_size = 2, strides = 2)])
    
            # generate pooled output
            output11 = modelll.predict(image2)
            output11 = np.squeeze(output11)
            
            output12=np.copy(output11)
            
            for i in range(output12.shape[0]):
                for j in range(output12.shape[1]):
                    
                    if output12[i][j]>1/4:
                        output12[i][j]=1
                    else:
                        output12[i][j]=0 
                        
    ############ APPLY KERNEL ############
                        
                        
            # kernel1 = [[[[1/9]],[[2/9]]],
            #           [[[2/9]],[[4/9]]]]
            
            # kernel2 = [[[[2/9]],[[1/9]]],
            #           [[[4/9]],[[2/9]]]]
            
            # output17,output18=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(2,2),kernel=kernel1)    #sobre las predicciones binarias 50%
            # output19,output20=filtro_Cov2_size_kernel(image=output18,size_kernel=(3,3),stride_x_y=(1,1),kernel=kernel2)   #sobre las predicciones binarias del % fijado    
            
            
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
            
            
            kernel1 = [[[[1/9]],[[2/9]]],
                      [[[2/9]],[[4/9]]]]
            
            kernel2 = [[[[2/9]],[[1/9]]],
                      [[[4/9]],[[2/9]]]]
            
            kernel3 = [[[[2/9]],[[4/9]]],
                      [[[1/9]],[[2/9]]]]
            
            kernel4 = [[[[4/9]],[[2/9]]],
                      [[[2/9]],[[1/9]]]]
            
            
            output17_a,output18_a=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel1)    #sobre las predicciones binarias 50%
            output17_b,output18_b=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel2)    #sobre las predicciones binarias 50%
            output17_c,output18_c=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel3)    #sobre las predicciones binarias 50%
            output17_d,output18_d=filtro_Cov2_size_kernel(image=image,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel4)    #sobre las predicciones binarias 50%

            
            output17=np.ones((output17_a.shape[0],output17_a.shape[1]))
            
            for filas in range(output17.shape[0]):
                for columnas in range(output17.shape[1]): 
                    
                    if filas%2==0 and columnas%2==0:
                        output17[filas][columnas]=output18_a[filas][columnas]
                    
                    if filas%2==0 and columnas%2==1:
                        output17[filas][columnas]=output18_b[filas][columnas]
                        
                    if filas%2==1 and columnas%2==0:
                        output17[filas][columnas]=output18_c[filas][columnas]
                        
                    if filas%2==1 and columnas%2==1:
                        output17[filas][columnas]=output18_d[filas][columnas]
                        

                
            

            output17_reshape = output17.reshape(1, output17.shape[0], output17.shape[1], 1)

            output19 = modelll.predict(output17_reshape)
            
            output19 = np.squeeze(output19)
            
            output20=np.copy(output19)
            
            for i in range(output20.shape[0]):
                for j in range(output20.shape[1]):
                    
                    if output20[i][j]>2/4:
                        output20[i][j]=1
                    else:
                        output20[i][j]=0 
            
            

            output21_a,output22_a=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel1)    #sobre las predicciones binarias %
            output21_b,output22_b=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel2)    #sobre las predicciones binarias %
            output21_c,output22_c=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel3)    #sobre las predicciones binarias %
            output21_d,output22_d=filtro_Cov2_size_kernel(image=image1,size_kernel=(2,2),stride_x_y=(1,1),kernel=kernel4)    #sobre las predicciones binarias %     
            
            
            output21=np.ones((output21_a.shape[0],output21_a.shape[1]))
            
            for filas in range(output21.shape[0]):
                for columnas in range(output21.shape[1]): 
                    
                    if filas%2==0 and columnas%2==0:
                        output21[filas][columnas]=output22_a[filas][columnas]
                    
                    if filas%2==0 and columnas%2==1:
                        output21[filas][columnas]=output22_b[filas][columnas]
                        
                    if filas%2==1 and columnas%2==0:
                        output21[filas][columnas]=output22_c[filas][columnas]
                        
                    if filas%2==1 and columnas%2==1:
                        output21[filas][columnas]=output22_d[filas][columnas]
            
            
            output21_reshape = output21.reshape(1, output21.shape[0], output21.shape[1], 1)
            output23 = modelll.predict(output21_reshape)
            output23 = np.squeeze(output23)
            
            output24=np.copy(output23)
            
            for i in range(output24.shape[0]):
                for j in range(output24.shape[1]):
                    
                    if output24[i][j]>2/4:
                        output24[i][j]=1
                    else:
                        output24[i][j]=0 
            
            
    # Color map of cancer 
    
            #Con esta salida voy a calcular el mapa de la imagen, para ello multiplicaré la pred de cada esquina por 50 y la localizaré en la posición correspondiente. 
            #utilizo la salido del filtro 2x2,1 por eso multiplico por 50, es como si cada parche lo dividiera en 4 sub_parches de 50x50
    
            img7_=np.ones(((max_y)*100,(max_x)*100,3))
            x=50
            
            #puede ser que mañana lo tenga que cambiar porque las imagenes de validacion las tengo ordenadas distintas a las de test en los ejes
            
            for i in tqdm(range(max_y2)):    #recorro las dimensiones de 50 en 50 
                for j in range(max_x2):
            
                    if output6[i][j]<0.5:    # si en cada filtrado de 2x2 del averagepooling me da menor que 0.5 quiere decir que hay más parches sanos que con tumor por lo que pongo la imagen original
            
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,:]=img2_[i*x:(i+1)*x,j*x:(j+1)*x,:]
                    else: # en caso contrario pongo la predicion multiplicada por 255 para que me indique la zona
            
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,0]=[255]
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,1]=[1-output6[i][j]*255]
                        img7_[i*x:(i+1)*x,j*x:(j+1)*x,2]=[0]
            
            
            etiquetas=[]      
            img8_=img7_.astype(np.uint8) 
            
            
            dst3 = cv2.addWeighteddst = cv2.addWeighted(img2_,0.7,img8_,0.3,0)     
    
    
            
            
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2')    
            # matplotlib.pyplot.imshow(output)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 (salida etiquetas tumor and normal)')    
            # matplotlib.pyplot.imshow(output2)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%')    
            # matplotlib.pyplot.imshow(output3)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%'+' (salida etiquetas tumor and normal)')   
            # matplotlib.pyplot.imshow(output4)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 con kernel')    
            # matplotlib.pyplot.imshow(output13)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 (salida etiquetas tumor and normal) con kernel')   
            # matplotlib.pyplot.imshow(output14)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'% con kernel')    
            # matplotlib.pyplot.imshow(output15)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 3x3,2 - '+str(limite_pred)+'%'+' (salida etiquetas tumor and normal) con kernel')   
            # matplotlib.pyplot.imshow(output16)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1')    
            # matplotlib.pyplot.imshow(output6)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2')    
            # matplotlib.pyplot.imshow(output7)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 (salida etiquetas tumor and normal)')   
            # matplotlib.pyplot.imshow(output8)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - ' +str(limite_pred)+'%')  
            # matplotlib.pyplot.imshow(output11)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - ' +str(limite_pred)+'%'+' round (al 50%)')   
            # matplotlib.pyplot.imshow(output12)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 con kernel')    
            # matplotlib.pyplot.imshow(output19)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 (salida etiquetas tumor and normal) con kernel')   
            # matplotlib.pyplot.imshow(output20)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - '+str(limite_pred)+'% con kernel')    
            # matplotlib.pyplot.imshow(output23)
            # plt.show()
            # matplotlib.pyplot.title('Overlapping y averagepooling 2x2,1 + 2x2,2 - '+str(limite_pred)+'% (salida etiquetas tumor and normal) con kernel')   
            # matplotlib.pyplot.imshow(output24)
            # plt.show()
            
            # matplotlib.pyplot.title('Mapa de calor Overlapping y averagepooling filtrado 2x2,1')      
            # matplotlib.pyplot.imshow(dst3)
            # plt.show()
    
            
    
            output2_parches_aislados=elimina_parches_aislados(output2)
            output4_parches_aislados=elimina_parches_aislados(output4)
            output8_parches_aislados=elimina_parches_aislados(output8)
            output12_parches_aislados=elimina_parches_aislados(output12)
            output14_parches_aislados=elimina_parches_aislados(output14)
            output16_parches_aislados=elimina_parches_aislados(output16)
            output20_parches_aislados=elimina_parches_aislados(output20)
            output24_parches_aislados=elimina_parches_aislados(output24)

            
            
            
            labels_real_2_1=aa.flatten()
            label_preds_overlapping_1_1 = output2.flatten()
            label_preds_overlapping_2_1 = output4.flatten()
            label_preds_overlapping_3_1 = output8.flatten()
            label_preds_overlapping_4_1 = output12.flatten()
            label_preds_overlapping_5_1 = output14.flatten()
            label_preds_overlapping_6_1 = output16.flatten()
            label_preds_overlapping_7_1 = output20.flatten()
            label_preds_overlapping_8_1 = output24.flatten()
            
            labels_real_2_1=aa.flatten()
            label_preds_overlapping_1_1_parches_aislados = output2_parches_aislados.flatten()
            label_preds_overlapping_2_1_parches_aislados = output4_parches_aislados.flatten()
            label_preds_overlapping_3_1_parches_aislados = output8_parches_aislados.flatten()
            label_preds_overlapping_4_1_parches_aislados = output12_parches_aislados.flatten()
            label_preds_overlapping_5_1_parches_aislados = output14_parches_aislados.flatten()
            label_preds_overlapping_6_1_parches_aislados = output16_parches_aislados.flatten()
            label_preds_overlapping_7_1_parches_aislados = output20_parches_aislados.flatten()
            label_preds_overlapping_8_1_parches_aislados = output24_parches_aislados.flatten()
            
            
            label_preds_overlapping_1_1_parches_aislados = output2_parches_aislados.flatten()
            label_preds_overlapping_3_1_parches_aislados = output8_parches_aislados.flatten()
            label_preds_overlapping_5_1_parches_aislados = output14_parches_aislados.flatten()
            label_preds_overlapping_7_1_parches_aislados = output20_parches_aislados.flatten()
            
            
            label_preds_binarias_sin_blancos,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, etiquetas_pred_binarias, labels_real_2_1)

            
            label_preds_overlapping_1_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_1_1, labels_real_2_1)
            label_preds_overlapping_2_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_2_1, labels_real_2_1)
            label_preds_overlapping_3_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_3_1, labels_real_2_1)
            label_preds_overlapping_4_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_4_1, labels_real_2_1)
            label_preds_overlapping_5_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_5_1, labels_real_2_1)
            label_preds_overlapping_6_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_6_1, labels_real_2_1)
            label_preds_overlapping_7_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_7_1, labels_real_2_1)
            label_preds_overlapping_8_1_1,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_8_1, labels_real_2_1)


            label_preds_overlapping_1_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_1_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_2_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_2_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_3_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_3_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_4_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_4_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_5_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_5_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_6_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_6_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_7_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_7_1_parches_aislados, labels_real_2_1)
            label_preds_overlapping_8_1_1_parches_aislados,labels_real_2_1_sin_blancos = elimina_parches_blancos_etiquetas(x_test_sin_overlapping_imagen_seleccionada, label_preds_overlapping_8_1_parches_aislados, labels_real_2_1)

            etiquetas_pred_binarias_sin_blancos_1=etiquetas_pred_binarias_sin_blancos
            etiquetas_pred_binarias_sin_blancos_append.extend(etiquetas_pred_binarias_sin_blancos_1)


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
            
            
            label_preds_overlapping_parches_aislados = label_preds_overlapping_1_1_1_parches_aislados
            label_preds_overlapping_2_parches_aislados = label_preds_overlapping_2_1_1_parches_aislados
            label_preds_overlapping_3_parches_aislados = label_preds_overlapping_3_1_1_parches_aislados
            label_preds_overlapping_4_parches_aislados = label_preds_overlapping_4_1_1_parches_aislados
            label_preds_overlapping_5_parches_aislados = label_preds_overlapping_5_1_1_parches_aislados
            label_preds_overlapping_6_parches_aislados = label_preds_overlapping_6_1_1_parches_aislados
            label_preds_overlapping_7_parches_aislados = label_preds_overlapping_7_1_1_parches_aislados
            label_preds_overlapping_8_parches_aislados = label_preds_overlapping_8_1_1_parches_aislados
            
            
    
            label_preds_overlapping_append.extend(label_preds_overlapping) 
            label_preds_overlapping_append_parches_aislados.extend(label_preds_overlapping_parches_aislados)                     
            
            
            label_preds_overlapping_2_append.extend(label_preds_overlapping_2) 
            label_preds_overlapping_2_append_parches_aislados.extend(label_preds_overlapping_2_parches_aislados) 
                      
    
            label_preds_overlapping_5_append.extend(label_preds_overlapping_5) 
            label_preds_overlapping_5_append_parches_aislados.extend(label_preds_overlapping_5_parches_aislados) 
            
            
            label_preds_overlapping_6_append.extend(label_preds_overlapping_6) 
            label_preds_overlapping_6_append_parches_aislados.extend(label_preds_overlapping_6_parches_aislados) 
            
            
            label_preds_overlapping_3_append.extend(label_preds_overlapping_3) 
            label_preds_overlapping_3_append_parches_aislados.extend(label_preds_overlapping_3_parches_aislados)      
        
            
            label_preds_overlapping_4_append.extend(label_preds_overlapping_4) 
            label_preds_overlapping_4_append_parches_aislados.extend(label_preds_overlapping_4_parches_aislados) 
                         
    
            label_preds_overlapping_7_append.extend(label_preds_overlapping_7)            
            label_preds_overlapping_7_append_parches_aislados.extend(label_preds_overlapping_7_parches_aislados) 
            
            
            label_preds_overlapping_8_append.extend(label_preds_overlapping_8) 
            label_preds_overlapping_8_append_parches_aislados.extend(label_preds_overlapping_8_parches_aislados) 
            
            
            
    ################################## Almacenamiento de datos en Txt 
            
            z=z+max_x*max_y
            z1=z1+max_x2*max_y2
    
    
    
    etiquetas_reales_sin_blanco_append=np.array(etiquetas_reales_sin_blanco_append)
    etiquetas_pred_binarias_append=np.array(etiquetas_pred_binarias_append)
    etiquetas_pred_binarias_porcentaje_fijado_append=np.array(etiquetas_pred_binarias_porcentaje_fijado_append)
    etiquetas_reales2_append=np.array(etiquetas_reales2_append)
    etiquetas_pred_binarias2_append=np.array(etiquetas_pred_binarias2_append)
    etiquetas_pred_binarias_porcentaje_fijado2_append=np.array(etiquetas_pred_binarias_porcentaje_fijado2_append)
    
    etiquetas_pred_binarias_sin_blancos_append=np.array(etiquetas_pred_binarias_sin_blancos_append)

    
    
    label_preds_overlapping_append=np.array(label_preds_overlapping_append)
    label_preds_overlapping_2_append=np.array(label_preds_overlapping_2_append)
    label_preds_overlapping_3_append=np.array(label_preds_overlapping_3_append)
    label_preds_overlapping_4_append=np.array(label_preds_overlapping_4_append)
    label_preds_overlapping_5_append=np.array(label_preds_overlapping_5_append)
    label_preds_overlapping_6_append=np.array(label_preds_overlapping_6_append)
    label_preds_overlapping_7_append=np.array(label_preds_overlapping_7_append)
    label_preds_overlapping_8_append=np.array(label_preds_overlapping_8_append)

    
    label_preds_overlapping_append_parches_aislados=np.array(label_preds_overlapping_append_parches_aislados)
    label_preds_overlapping_2_append_parches_aislados=np.array(label_preds_overlapping_2_append_parches_aislados)
    label_preds_overlapping_3_append_parches_aislados=np.array(label_preds_overlapping_3_append_parches_aislados)
    label_preds_overlapping_4_append_parches_aislados=np.array(label_preds_overlapping_4_append_parches_aislados)
    label_preds_overlapping_5_append_parches_aislados=np.array(label_preds_overlapping_5_append_parches_aislados)
    label_preds_overlapping_6_append_parches_aislados=np.array(label_preds_overlapping_6_append_parches_aislados)
    label_preds_overlapping_7_append_parches_aislados=np.array(label_preds_overlapping_7_append_parches_aislados)
    label_preds_overlapping_8_append_parches_aislados=np.array(label_preds_overlapping_8_append_parches_aislados)    
    
    
    etiquetas_pred_binarias_sin_blancos_total.extend(etiquetas_pred_binarias_sin_blancos_append)

    
    a=classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normalfloat,tumorfloat,accuracy_float)
    
    BACC_vector= np.append(BACC_float, balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append))
    MCC_vector= np.append(MCC_float,matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_append))

    
    a=classification_report(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(etiquetas_reales_sin_blanco_append, etiquetas_pred_binarias_porcentaje_fijado_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_append))
    

    a=classification_report(labels_real_2_append, label_preds_overlapping_2_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_2_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_2_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_5_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_5_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_5_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_6_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_6_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_6_append))
    

    a=classification_report(labels_real_2_append, label_preds_overlapping_3_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_3_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_3_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_4_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_4_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_4_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_7_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_7_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_7_append))
    
    
    a=classification_report(labels_real_2_append, label_preds_overlapping_8_append, target_names=classes2)
    normal_float_vector,tumor_float_vector,accuracy_vector=organiza_datos(a,normal_float_vector,tumor_float_vector,accuracy_vector)
    
    BACC_vector= np.append(BACC_vector, balanced_accuracy_score(labels_real_2_append, label_preds_overlapping_8_append))
    MCC_vector= np.append(MCC_vector,matthews_corrcoef(labels_real_2_append, label_preds_overlapping_8_append))
    
    
    

    
    tablas_tumor['split_'+str(split)+'_tumor'] = tumor_float_vector
    tablas_normal['split_'+str(split)+'_normal'] = normal_float_vector
    tablas_accuracy['split_'+str(split)+'_accuracy'] = accuracy_vector
    tablas_BACC['split_'+str(split)+'_BACC'] = BACC_vector
    tablas_MCC['split_'+str(split)+'_MCC'] = MCC_vector


        
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

    label_preds_overlapping_total_parches_aislados.extend(label_preds_overlapping_append_parches_aislados)
    label_preds_overlapping_2_total_parches_aislados.extend(label_preds_overlapping_2_append_parches_aislados)
    label_preds_overlapping_3_total_parches_aislados.extend(label_preds_overlapping_3_append_parches_aislados)
    label_preds_overlapping_4_total_parches_aislados.extend(label_preds_overlapping_4_append_parches_aislados)
    label_preds_overlapping_5_total_parches_aislados.extend(label_preds_overlapping_5_append_parches_aislados)
    label_preds_overlapping_6_total_parches_aislados.extend(label_preds_overlapping_6_append_parches_aislados)
    label_preds_overlapping_7_total_parches_aislados.extend(label_preds_overlapping_7_append_parches_aislados)
    label_preds_overlapping_8_total_parches_aislados.extend(label_preds_overlapping_8_append_parches_aislados)
  


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

  
plt.show()
plot_confusion_matrix(etiquetas_reales_total, etiquetas_pred_binarias_sin_blancos_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='CNN' ) 

matplotlib.pyplot.title('CNN')   
plt.show()
            
            
plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method1' ) 

matplotlib.pyplot.title('method1')   
plt.show()

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method1_parches_aislados' ) 

matplotlib.pyplot.title('method1_parches_aislados')   
plt.show()
            


plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_2_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method1 60%' ) 

matplotlib.pyplot.title('method1 60%')   
plt.show()

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_2_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method1_parches_aislados 60%' ) 

matplotlib.pyplot.title('method1_parches_aislados 60%')   
plt.show()
            


plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_5_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method2' ) 

matplotlib.pyplot.title('method2')   
plt.show()

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_5_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method2_parches_aislados' ) 

matplotlib.pyplot.title('method2_parches_aislados')   
plt.show()
            

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_6_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method2 60%') 

matplotlib.pyplot.title('method2 60%')   
plt.show()

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_6_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method2_parches_aislados 60%' ) 

matplotlib.pyplot.title('method2_parches_aislados 60%')   
plt.show()




plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_3_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method3' ) 

matplotlib.pyplot.title('method3')   
plt.show()
            

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_3_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method3_parches_aislados' ) 


plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_4_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method3 60%' ) 

matplotlib.pyplot.title('method3 60%')   
plt.show()
            

plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_4_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method3_parches_aislados 60%' ) 

matplotlib.pyplot.title('method3_parches_aislados 60%')   
plt.show()




plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_7_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method4' ) 

matplotlib.pyplot.title('method4')   
plt.show()

plt.show()

plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_7_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method4_parches_aislados' ) 

matplotlib.pyplot.title('method4_parches_aislados')   
plt.show()


plt.show()
plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_8_total,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method4 60%' ) 

matplotlib.pyplot.title('method4 60%')   
plt.show()

plt.show()

plot_confusion_matrix(etiquetas_reales_total, label_preds_overlapping_8_total_parches_aislados,
                      classes2,
                      save_name='CM_.png',
                      normalize=False,
                      title='method4_parches_aislados 60%' ) 

matplotlib.pyplot.title('method4_parches_aislados 60%')   
plt.show()





