

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:59:35 2022

@author: sergi
"""

import cv2
import pandas as pd
import numpy as np 
import random 
import glob 
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn import preprocessing
from keras import backend as K
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import math
from numpy.random import seed
from skimage import filters

from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import (Dense, GlobalAveragePooling2D,
                          MaxPooling2D, Conv2D, Input, Flatten,
                          Dropout)
from keras.initializers import glorot_normal
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import cv2
from sklearn.metrics import classification_report
from plot_conf_matrix_and_visulaize_result import plot_confusion_matrix,visualize_results
from utils import curve_auc
import keras 


    
# WiDTH and HEIGHT
width = 100
height = 100
# PATHS AND VARIABLES
splits = 5 # number of segments we have created
#path_input = '../Patch/inputs/inputs_labels_split_path__distribucion_3400_'

classes = ['tumor', 'normal'] # name of the classes
# CNN hyper-parameters
epochs = 50  # number of epochs for training
lr = 0.001 # value of initial learning rate
batch_size = 512 # value of batch size
output_layer=2
lr = [0.01,0.001,0.0001] # value of learning rate

Optimizers_name=['SGD','Adam']   # two optimizers
Optimizers=[SGD,Adam]

porcentaje_entrenamiento=[0,30,70,50]  # percentage trained layers

tra=['Non_trainable','Trainable','trainable','trainable','trainable']

architecture=[DenseNet121,InceptionV3,ResNet50,VGG16,ResNet101]
architecture_name=['DenseNet121','InceptionV3','ResNet50','VGG16','ResNet101']



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):   #Decay function of the lr
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate




#for j in range(np.size(lr)):
for j in range(2,3):                            # j fix the architecture   (with this evaluate two architectures)
    
#    for i in range(np.size(Optimizers)):       # i fix the optimizer  (may vary according to needs)
    for i in range(2):
        
      for w in range(1):                        # w fix the percentage of layer that are trained

        #Create the variables
        label_pre_total=[]
        label_tru_total=[]
        label_test_total=[]
        label_tru_total_test=[]
        Accuracy_train_mean=np.empty(shape=splits)
        Accuracy_test_mean=np.empty(shape=splits)
        range_accuracy_train_mean=np.empty(shape=splits)
        Accuracy_validation_mean=np.empty(shape=splits)
        range_accuracy_validation_mean=np.empty(shape=splits)
        
        print('Con balanceo 1_1_4 pero con data augmentation ')
        
        for split in range(5):   #number of split the data

            val_acc = []
                  
                
            print("Split {}/{}".format(split, splits))
            print("Loading data")
            # print('Learning rate = '+str(lr[j]))
            print('Optimizers = '+Optimizers_name[i])
            print('The CNN used is:'+str(architecture[j]))

            # The inputs are divided into train and validation
            inputs = np.load('C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/inputs_train_test_validation/'+'inputs_labels_split_path_train_validation_split_split_'+str(split)+'_100x100.npy',  # loads the .npy file where the patches of the different splits are located.
                            allow_pickle=True).item()
            
            #inputs = np.load('C:/Users/Sergio/Documents/CANCER/inputs4/inputs_labels_split_path_2-100x100_rgb_15_porc_blanco_220_sin_14_test_2.npy', 
              #                allow_pickle=True).item()
            
            print("Finished loading data")
            #evaluate de value and the label of the train and validation data
            x_train = np.asarray(inputs['x_train'])
            y_train = inputs['y_train']
            x_val = np.asarray(inputs['x_val'])
            y_val = inputs['y_val']
        
        
            # Label Encoder
            # The tag values must be left in one-hot-encoding format so that the network can be trained correctly.
            # so that the network can be trained correctly. This is necessary for the
            # calculation of the loss function.
          
            le = preprocessing.LabelEncoder()
            le.fit(classes)
            y_train = le.transform(list(y_train))
            #y_test = le.transform(list(y_test))
            y_val = le.transform(list(y_val))
        
            # Image normalization
            # Due to the matrix calculations that are performed in the training of the network, it helps to 
            # Due to the matrix calculations that are performed in the training of the network, it helps if the values are in the range 0-1.
            # As the maximum value of a pixel is 255, we divide the value of each pixel in each channel by 255.
            # the value of each pixel in each channel by 255.
          
            x_train = np.divide(x_train, 255)
            #x_test = np.divide(x_test, 255)
            x_val = np.divide(x_val, 255)
        
            y_train = to_categorical(y_train,num_classes=2)
            #y_test = to_categorical(y_test)
            y_val = to_categorical(y_val,num_classes=2)
            
            # to increase the number of samples we made changes to the pictures
          
        datagen = ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                vertical_flip=True,
                #fill_mode='constant',
                #brightness_range=[0.7,1],
                zoom_range=[0.5,1.0],
                cval=1)# prepare iterator
            
            
            datagen.fit(x_train)  #This will compute the statistics required to 
                                  #perform the transformations on your image data
        
            # Model declaration
            # We declare the model we are going to use. In this case it will be InceptionV3
            # using the weights pre-entered in the Imagenet dataset.
            # Then we add some layers that will be the ones that we will use to train
            # as a classifier with the features extracted by the layers. 
            # convolutional
        
            input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
            
            if w==0: # we make the base model untrained
                
                base_model = architecture[j](weights='imagenet', include_top=False,
                                          input_shape=input_shape)
                base_model.trainable = False 
              
            elif w!=0: #we train a percentage of the layers of the model
                
                base_model = architecture[j](weights='imagenet', include_top=False,
                                          input_shape=input_shape)
                
                
                for lay in range(len(base_model.layers[:])):
                    if lay<int(len(base_model.layers[:])*(1-porcentaje_entrenamiento[w]*1/100)):
                        base_model.layers[lay].trainable = False
                    else:
                        base_model.layers[lay].trainable = True

# we add to the base model a number of necessary layers
            x = base_model.output
            x = Flatten()(x)
            x = Dropout(0.5)(x)
            x = Dense(512, activation='relu',
                      kernel_initializer=glorot_normal(seed=99))(x)
            preds = Dense(output_layer, activation='softmax',
                          kernel_initializer=glorot_normal(seed=99))(x)
            model = Model(inputs=base_model.input, outputs=preds)
        
            Train_count=0
            for layer in model.layers:
                salida=layer.trainable
                
                if salida==True:
                    Train_count=Train_count+1
            
            print('El numero de capas que se entrenan son:'+str(Train_count))
            print('El % de capas que se entrenan :'+str((Train_count/len(model.layers))*100)+' %')

            # Optimizer
            # The optimiser is in charge of training the network. It will use as
            # input the value of the loss and it will determine where it should move the
            # weights. The learning rate value determines how far it should follow the # direction of the gradient.
            # the direction of the gradient. A small lr value will make the # training take a long time, because 
            # training takes too long, because we vary the weights too little. However
            # will ensure that we do not go out of an optimum in the search space. With a larger lr
            # training will be faster but we may overshoot local or global optima.
            # local or global optima.
            
            #optimizer = Optimizers[i](learning_rate=lr[j])
            

            momentum=0.8
            if i==0:
                optimizer = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False) 
                loss_history = LossHistory()
                lrate = LearningRateScheduler(step_decay)
                callbacks_list = [loss_history, lrate]

           
            elif i==1:
                
                optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

                # initial_learning_rate = 0.1
                # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                #     initial_learning_rate,
                #     decay_steps=1000,
                #     decay_rate=0.96,
                #     staircase=True)

                # optimizer = Adam(learning_rate=lr_schedule)
            
            # Early Stopping
            # EAS is a technique used to prevent overfitting in a network.
            # It monitors the error in validation. If over several epochs
            # loss in training keeps going down but in the validation set it goes up, stop the training and return to the 
            # validation set goes up, the training is stopped and returns to the weights of the 
            # weights of the network at the time when this event started to occur.
            eas = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10,
                                verbose=0,
                                mode='auto', restore_best_weights=True)
            
            # Model compilation and fitting
            # With the compile function we define the type of optimiser, the loss function and which metrics we want to display during training.
            # and which metrics we want to visualise during training.
            model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                          metrics=['accuracy'])    
        
        
            
            # With the fit_generator function we train the model. Datagen.flow performs the data augmentation operations
            # performs the data augmentation operations we have described
            # above. We pass it the number of epochs and the validation set.
            # In callbacks we pass the Early Stopping function that was defined # earlier. 
            # above.
            history = model.fit_generator(
                                datagen.flow(x_train, y_train, 
                                              batch_size=batch_size),
                                verbose=1,
                                epochs=epochs,
                                validation_data=(x_val, y_val),
                                #callbacks = [eas],
                                callbacks=callbacks_list,
                                shuffle=True)
          
        
            val_preds = model.predict(x_val)
            val_preds = to_categorical(np.argmax(val_preds, axis=1),num_classes=2)
            val_acc.append(accuracy_score(y_val, val_preds))
            print("Val acc: {}".format(val_acc[-1]))
        
        
            plot_confusion_matrix(y_val.argmax(axis=1), val_preds.argmax(axis=1),
                                  classes,
                                  save_name='CM-' + str(split) +'-'+str(tra[w])+'lr-' +'-'+ str(Optimizers_name[i])+ '.png',
                                  normalize=False,
                                  title='Val Split' + str(split))
            
            # If desired, we can save the weights of the network in that split
            # for later use.
            # model.save('nombredemimodelo.h5')
        
            print(classification_report(y_val.argmax(axis=1), val_preds.argmax(axis=1), target_names=classes))

            label_pre_total=np.concatenate((label_pre_total,val_preds.argmax(axis=1)), axis=None)
            label_tru_total=np.concatenate((label_tru_total,y_val.argmax(axis=1)), axis=None)
    

            #save de data of model
            if w==0: 
                
                model.save('C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/modelos/model_'+architecture_name[j]+'_split_'+str(split)+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+'_non_trainable_optimiz_100x100_inputs_labels_split_path_0-100x100.h5')
                newfile = open("C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/salida_train_validation_test_sin_overlapping/CNN_training_and_testeo_model_"+
                               architecture_name[j]+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+"_non_trainable_balance_menor.txt", "a+")

            elif w!=0:
                model.save('C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/modelos/model_'+architecture_name[j]+'_split_'+str(split)+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+'_trainable_'+str(porcentaje_entrenamiento[w])+'%_optimiz_100x100_inputs_labels_split_path_0-100x100.h5')
                newfile = open("C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/salida_train_validation_test_sin_overlapping/CNN_training_and_testeo_model_"+
                               architecture_name[j]+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+'_trainable_'+str(porcentaje_entrenamiento[w])+"%_balance_menor.txt", "a+")


        # PLOT OF THE RESULT IN SCREEN AND SAVE IN A FILE
        
            fig = plt.figure()
            epoch_values = list(range(epochs))
            plt.plot(epoch_values, history.history['loss'], label='Pérdida de entrenamiento')
            plt.plot(epoch_values, history.history['val_loss'], label='Pérdida de validación')
            plt.plot(epoch_values, history.history['accuracy'], label='Exactitud de entrenamiento')
            plt.plot(epoch_values, history.history['val_accuracy'], label='Exactitud de validación')
            plt.title('Pérdida y Exactitud de Entrenamiento')
            plt.xlabel('Epoch N°')
            plt.ylabel('Pérdida/Exactitud')
            plt.legend()  
            plt.close(fig)

            
            
            fig = plt.figure()
            plt.plot(range(1,epochs+1),loss_history.lr,label='learning rate')
            plt.xlabel("epoch")
            plt.xlim([1,epochs+1])
            plt.ylabel("learning rate")
            plt.legend(loc=0)
            plt.grid(True)
            plt.title("Learning rate")
            plt.show()
            plt.close(fig)
            

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = len(acc)
            
            conf_matrix=confusion_matrix(y_val.argmax(axis=1), val_preds.argmax(axis=1))
            
            Accuracy_train_mean[split]=acc[-1]
            range_accuracy_train_mean[split]=np.std(acc)
            
            Accuracy_validation_mean[split]=val_acc[-1]
            range_accuracy_validation_mean[split]=np.std(val_acc)
                    


            newfile.write("New iteration CNN  \n \n")
            newfile.write(str(architecture[j])+'\n \n ')
            newfile.write("SPLIT : "+ str(split) +'\n \n')

            newfile.write("Learning rate variable"+'\n')
            newfile.write("Optimizers"+ Optimizers_name[i] +'\n')
            newfile.write('Numbers of Total layers that are trained:'+ str(Train_count)+'\n')
            newfile.write('Percentage of Total layers that are trained:' + str((Train_count/len(model.layers))*100)+' %'+'\n'+'\n')

            
            #newfile.write("CNN name:"+ str())
            newfile.write('Accuracy in val output: '+str(val_acc[-1]) +'\n \n')
            newfile.write(classification_report(y_val.argmax(axis=1), val_preds.argmax(axis=1), target_names=classes))
            newfile.write("Mean accuracy in Val: {} +- {}".format(np.mean(val_acc),
                                                          np.std(val_acc))+'\n'+'\n')
        
            
            newfile.write(str(conf_matrix)+'\n'+'\n')
            
            newfile.write('Summary all Epoch:'+'\n')
            newfile.write('Training accuracy:'+'\n')
            newfile.write(str(acc)+'\n')
            newfile.write('Validation accuracy:'+'\n')
            newfile.write(str(val_acc)+'\n')
            newfile.write('Training loss:'+'\n')
            newfile.write(str(loss)+'\n')
            newfile.write('Validation loss:'+'\n')
            newfile.write(str(val_loss)+'\n')    
            newfile.write('Number of epochs:'+'\n')
            newfile.write(str(epochs)+'\n')
            
            newfile.write('\n'+'\n'+'\n'+'\n')
            
            
            classes = ['tumor', 'normal']

            inputs = np.load('C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/inputs_train_test_validation/'+'inputs_labels_split_path_test_sin_overlapping_split_'+str(split)+'_100x100.npy', 
                            allow_pickle=True).item()

            #inputs = np.load('D:/Sergio/Patch/Data_6_4/inputs3/inputs_labels_test_4_img_test_100x100_rgb.npy',
            #                             allow_pickle=True).item()

                        
            print("Finished loading data")
            x_test = np.asarray(inputs['x_test'])
            y_test = inputs['y_test']
            name_images=inputs['Name_imagen']
                    
                    
                        # Label Encoder
                        # Los valores de las etiquetas hay que dejarlos en formato one-hot-encoding
                        # para que se pueda entrenar la red correctamente. Es necesario para el
                        # cálculo de la función de pérdida.
            le = preprocessing.LabelEncoder()
            le.fit(classes)
            y_test = le.transform(list(y_test))
                        #y_test = le.transform(list(y_test))

                    
                        # Image normalization
                        # Debido a los cálculos matriciales que se realizan en el entrenamiento de 
                        # la red, que los valores se encuentren en un rango de 0-1 ayuda al mismo.
                        # Como el máximo valor de un pixel es 255, es por ello que dividimos
                        # el valor de cada pixel en cada canal por 255.
            x_test = np.divide(x_test, 255)
                    
            y_test = to_categorical(y_test,num_classes=2)
            


            val_preds = model.predict(x_test)
            
            
            label_test_total=np.concatenate((label_test_total,val_preds.argmax(axis=1)), axis=None)
            label_tru_total_test=np.concatenate((label_tru_total_test,y_test.argmax(axis=1)), axis=None)

            etiquetas_val3=val_preds[:,1]
            etiquetas_mas_70=val_preds[:,1]

            val_preds = to_categorical(np.argmax(val_preds, axis=1),num_classes=2)
            acc=accuracy_score(y_test, val_preds)
            
            Accuracy_test_mean[split]=acc

            etiquetas_val=val_preds.argmax(axis=1)
            etiquetas_test=y_test.argmax(axis=1)
            print("Val acc: {}".format(acc))

            conf_matrix=confusion_matrix(etiquetas_test, etiquetas_val)


            plot_confusion_matrix(etiquetas_test, etiquetas_val,
                                              classes,
                                              save_name='CM_testeo.png',
                                              normalize=False,
                                              title='Val Split' )

            print(classification_report(etiquetas_test, etiquetas_val, target_names=classes))

            
            newfile.write("Testeo imagenes sin overlapping"+'\n'+'\n')
            newfile.write("Nombre de la imagenes: "+ str(name_images)+'\n'+'\n')
            
            
            #newfile.write("CNN name:"+ str())
            newfile.write(classification_report(etiquetas_test, etiquetas_val, target_names=classes))
            newfile.write("Test accuracy: {}".format(acc)+'\n'+'\n')
            
            newfile.write(str(conf_matrix)+'\n'+'\n')
            

            
            newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')            
            newfile.write('\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n')

            newfile.close()
            
            
            
            K.clear_session()



        if w==0:
                
            newfile = open("C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/salida_train_validation_test_sin_overlapping/CNN_training_and_testeo_model_"+
                              architecture_name[j]+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+"_non_trainable_balance_menor.txt", "a+")
            
        elif w!=0:      
            
            newfile = open("C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/salida_train_validation_test_sin_overlapping/CNN_training_and_testeo_model_"+
                              architecture_name[j]+'_'+str(epochs)+'_epochs_256_batchsize_'+Optimizers_name[i]+'_trainable_'+str(porcentaje_entrenamiento[w])+"%_balance_menor.txt", "a+")
            
            
            

        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n') 
                  
                 
        conf_matrix=confusion_matrix(label_tru_total, label_pre_total)

        newfile.write("RESUMEN TOTAL VALIDACION"+'\n'+'\n')
        #newfile.write("CNN name:"+ str())
        newfile.write("Mean accuracy in Val: {} +- {}".format(np.mean(val_acc),
                                                          np.std(val_acc))+'\n'+'\n')        
        newfile.write(classification_report(label_tru_total, label_pre_total, target_names=classes))

        
        newfile.write(str(conf_matrix)+'\n'+'\n')
        
        newfile.close()




        newfile = open("C:/Users/Sergio/Documents/CANCER/balanceo_1_1_4/salida_train_validation_test_sin_overlapping/RESUMEN_TOTAL_balance_menor.txt", "a+")
        
        newfile.write(" --------- CNN : "+ str(architecture[j])+'------------------ \n \n ')        

        newfile.write(" - 50 Epochs \n")
        newfile.write(" - Bath size 256 \n")
        newfile.write(" - Optimizers"+ Optimizers_name[i] +'\n')
        newfile.write(' - Numbers of Total layers that are trained:'+ str(Train_count)+'\n')
        newfile.write(' - Percentage of Total layers that are trained:' + str((Train_count/len(model.layers))*100)+' %'+'\n'+'\n')
        
        newfile.write("- RESUMEN ALL SPLITS"+'\n'+'\n')
            
            
        newfile.write("Mean accuracy in splits training: {} +- {}".format(np.mean(Accuracy_train_mean),
                                                              np.mean(range_accuracy_train_mean))+'\n'+'\n')  
        

        
            
        conf_matrix=confusion_matrix(label_tru_total, label_pre_total)

        newfile.write("RESUMEN TOTAL VALIDACION"+'\n'+'\n')
        
        
        newfile.write("Mean accuracy in splits validation : {} +- {}".format(np.mean(Accuracy_validation_mean),
                                                              np.mean(range_accuracy_validation_mean))+'\n'+'\n')      
        
        newfile.write(classification_report(label_tru_total, label_pre_total, target_names=classes))

        
        newfile.write(str(conf_matrix)+'\n'+'\n')
        
        
        conf_matrix=confusion_matrix(label_tru_total_test, label_test_total)

        newfile.write("RESUMEN TOTAL TEST"+'\n'+'\n')
    
        newfile.write(classification_report(label_tru_total_test, label_test_total, target_names=classes))

        
        newfile.write(str(conf_matrix)+'\n'+'\n')
        
        newfile.write("Mean accuracy in test: {} +- {}".format(np.mean(Accuracy_test_mean),
                                                              np.std(Accuracy_test_mean))+'\n'+'\n')   
        
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n')            
        newfile.write(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - '+'\n') 
        
        newfile.close()
        
