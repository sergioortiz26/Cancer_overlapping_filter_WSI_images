# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:47:43 2022

@author: sergio
"""

#Script to estract anda save patches



import cv2 
import matplotlib 
import glob
import numpy as np


path='C:/Users/Sergio/Documents/CANCER/HUP_images/'                                           # image folder path
path_mask='C:/Users/Sergio/Documents/CANCER/HUP_masks/'                                         # mask folder path          
#path_patch='C:/Users/Sergio/Documents/CANCER/Patch_100x100_train_and_validation/' 
path_patch='C:/Users/Sergio/Documents/CANCER/imagen_8917/'                                    # path to save the image patches



for i in glob.glob(path+'*.png'):
    
    name=str(i)
    f=name.replace('\\','/' )
    save_name = f.split('/')[-1]   #split()método divide una cadena en una lista
    print(save_name)
    mascara=f.split('/')[-1].split('_')[0] 
    img_bgr=cv2.imread(path+save_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #matplotlib.pyplot.imshow(img_rgb)
    
    for j in glob.glob(path_mask+mascara+'*'):

        name_mask=str(j)
        f=name_mask.replace('\\','/' )
        save_name_mask = f.split('/')[-1]   #split()método divide una cadena en una lista
        print(save_name_mask)
        
        img_mask=cv2.imread(path_mask+save_name_mask)
        #matplotlib.pyplot.imshow(img_mask)
    
    x=100

    hight =img_rgb.shape[0]
    width= img_rgb.shape[1]
        
    yy=int(hight/x)
    xx=int(width/x)
    
    
    patch=[]
    patch_mask=[]
    
    for k in range(int(xx)):
        for l in range(int(yy)):
            
            coord_min_y=l*x
            coord_max_y=(l+1)*x
            coord_min_x=k*x
            coord_max_x=(k+1)*x
            
            patch=img_rgb[coord_min_y:coord_max_y,coord_min_x:coord_max_x,:]
            #patch_resize = cv2.resize(patch,[299,299])
            
            patch_mask=img_mask[coord_min_y:coord_max_y,coord_min_x:coord_max_x,:]
            
            
            media_path=int(np.mean(patch))
            media_mask=int(np.mean(patch_mask))
            
            if  media_mask>38:    # 38 why is the 0.15% of 255 
                label='tumor'
            else:
                label='normal'    
                
            if k<10:
                coord_x='0'+str(k)
            else:
                coord_x=str(k)
                
            if l<10:
                coord_y='0'+str(l)
            else:
                coord_y=str(l)
                                
                
            if (np.mean(patch[:,:,0]) < 230.0 or np.mean(patch[:,:1]) < 230.0 or np.mean(patch[:,:,2]) < 230.0):
                
                cv2.imwrite(path_patch+mascara+'.'+coord_y+'.'+coord_x+'.'+label+'.png',patch)
            
            