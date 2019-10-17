# Pneumothorax-Classifier

Takes in chest radiographs with labels of -1 for no pneumothorax, and values for a mask of the pneumothorax, and it uses different model architectures to determine if out of sample chest radiographs contain pneumothorax and where. 

## Machine Learning Techniques used: 

#### Preprocessing Techniques: 
* Edge detection filtering
* Intensity Threshold to reduce noise
* Cropping radiograph images to focus on ribcage
* Image augmentation through translation, shearing, rotation, etc. 

#### Training Techniques
* Train dataset split by 50% positive and 50% negative
* Early stopping and saving best model during training
* Architectures include CNN and U-net
* Classification type includes binary and segmentation

#### Model Tuning Techniques
* Resampling ensemble
* K-fold cross validation
* Weighted averaging ensemble


## Future improvements
* Grid Search
* Transfer Learning
* Bagging


This project is being designed to be easy to add in a new 2D radiograph of another body part, and require minimal changes to the code. 
