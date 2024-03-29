#scrolls 50*100 patches on the mask and labels the area depending on a percentage. 
#If the mean of the mask is greater than 38 it labels it as a tumour because most of the patch is tumour (38 why is the 0.15% of 255).
#It performs an overlapping process as it does not change directly to the next patch but stays in the middle of the previous patch to have that overlapping. 

import cv2 
import matplotlib 
import glob
import numpy as np

path='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_imgs_idx5/CINJ_imgs_idx5/'
path_mask='C:/Users/Sergio/Documents/CANCER/doi_10.5061_dryad.1g2nt41__v1/CINJ_masks_HG/CINJ_masks_HG/'
path_patch='C:/Users/Sergio/Documents/CANCER/Patch_CINJ_test_overlapping_15_porc/'

# path='D:/Sergio/doi_10.5061_dryad.1g2nt41__v1/CINJ_imgs_idx5/CINJ_imgs_idx5/'
# path_mask='D:/Sergio/doi_10.5061_dryad.1g2nt41__v1/CINJ_masks_HG/CINJ_masks_HG/'
# path_patch='D:/Sergio/Patch/Data_6_4/patch_100_100_imagenes_CINJ/'


for i in glob.glob(path+'*.png'):
    
    name=str(i)
    f=name.replace('\\','/' )
    save_name = f.split('/')[-1]  
    print(save_name)
    mascara=f.split('/')[-1].split('_')[0] 
    img_bgr=cv2.imread(path+save_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #matplotlib.pyplot.imshow(img_rgb)
    
    for j in glob.glob(path_mask+mascara+'*'):

        name_mask=str(j)
        f=name_mask.replace('\\','/' )
        save_name_mask = f.split('/')[-1]   
        print(save_name_mask)
        
        img_mask=cv2.imread(path_mask+save_name_mask)
        #matplotlib.pyplot.imshow(img_mask)
    
    x=50
    y=100
    hight =img_rgb.shape[0]
    width= img_rgb.shape[1]
    
    yy=int(hight/x)
    xx=int(width/x)
                
    
    patch=[]
    patch_mask=[]
    
    for k in range(int(xx)-1):
        for l in range(int(yy)-1):
            
            coord_min_y=l*x
            coord_max_y=l*x+y
            coord_min_x=k*x
            coord_max_x=k*x+y
            
            patch=img_rgb[coord_min_y:coord_max_y,coord_min_x:coord_max_x,:]
            #patch_resize = cv2.resize(patch,[299,299])
            
            patch_mask=img_mask[coord_min_y:coord_max_y,coord_min_x:coord_max_x,:]
            contador=0
            
            media_path=int(np.mean(patch))
            media_mask=int(np.mean(patch_mask))
            
            if  media_mask>38:        # 38 why is the 0.15% of 255 
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
                                
                
            #if media_path<240:
                
            cv2.imwrite(path_patch+mascara+'.'+coord_y+'.'+coord_x+'.'+label+'.png',patch)
            
            


    
    
