# Cancer_overlapping_filter_WSI_images

Code for different filters in the test phase to improve label assignment by predicting a CNN.


Whole slide image (WSI) and the tumor mask associated with the normal tissue (zone black) and the tumor tissue (zone white). 

----------------------------------
CREATES PATCHES OF THE IMAGES WSI

Throught crea_parches_test_overlapping, crea_parches_test_sin_overlapping and crea_parches_train_and_validation are create the different patches asociated to the images. 
  - Crea_parches_train_and_validation: Create patches asociated with train and validate phase. 
  - crea_parches_test_sin_overlapping: Create patches associated with the test phase, patches that do not overlap with each other.
  - crea_parches_test_overlapping: Create patches associated with the test phase, patches that overlap with each other.
  
   To corroborate that each image has been well divided into patches, you can use the script comprobacion_mascaras.py, where selecting the path of the patches of an image, you can redo them to check their correct division into patches. 
   Note that this is not done with the training and validation images because at this stage the background patches that do not correspond to the tissue have been removed.
   
-------------------------------------------------
KFOLD, TRAINING AND VALIDATION PHASE

With Kfold_balanceo_1_1_4.py we split the dataset of patches (previously created) into different splits for training, validation, and testing.
For this process we proporcionated the path of patches with train and validation, with test with overlapping and the test without overlapping. In addition to this, the path of the images is provided, so as not to repeat patches of the same images in the 3 stages (training, validation and testing). 

One time that the patches are divided, traing different configuration to study the influence to use different achitectures, optimizers and percentage of layers training in the CNN. We use CNN_V2_s2_balanceo.py


---------------------------------------------
