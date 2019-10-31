# Pneumothorax-Classifier

Takes in chest radiographs with labels of -1 for no pneumothorax, and values for a mask of the pneumothorax, and it uses different model architectures to determine if out of sample chest radiographs contain pneumothorax and where. 




## Machine Learning Techniques used: 

#### Preprocessing Techniques: 
* Edge detection filtering
* Intensity Threshold to reduce noise
* Cropping radiograph images to focus on ribcage
* Image augmentation through translation, shearing, rotation, etc. 

#### Model Training/Tuning Techniques
* Train dataset split by 50% positive and 50% negative
* Early stopping and saving best model during training
* Architectures include CNN and U-net
* Classification type includes binary and segmentation
* Resampling ensemble
* K-fold cross validation
* Weighted averaging ensemble
* Bagging (Bootstrap Aggregation)


## Future improvements
* Add Grid Search for model training
* Add Transfer Learning for model training
* Add Horizontal Ensemble for model training
* Add Snapshot Ensemble for model training
* Add Stacked Ensemble for model training
* Add Testing option for existing model training sessions


This project is being designed to be easy to add in a new 2D radiograph of another body part, and require minimal changes to the code. 





#### Min Requirements (or latest versions):
* Python - 3.5.4
* Keras - 2.2.4
* Tensorflow - 1.8.0
* Pydicom - 1.3.0
* Imageio - 2.2.0
* Pillow - 4.2.1
* CV2 - 4.1.1
* Pandas - 0.22.0
* Scipy - 1.1.0
* Numpy - 1.13.3
