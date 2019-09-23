from DICOM_reader import DICOMReader
from DataHandler import DataHandler
from ImagePreprocessor import ImagePreprocessor
from DataGenerator import DataGenerator

from mask_functions import rle2mask

import sys
import os
import random

#ML libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import imageio

#For my windows machine
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


"""

Handles CNN training, validation, and testing

"""
class CNNClassifier:

    image_width = 512
    image_height = 512

    dicom_reader = None
    data_handler = None
    image_preprocessor = None

    def __init__(self):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()
        self.image_preprocessor = ImagePreprocessor()

    # #returns 1024x1024 raw pixel data of feature dicom images
    # # feature images will have pixel intensities
    # # target images will have binary denoting masks of pneumothorax
    # def get_unprocessed_feature_target_images(self, max_num=None):
    #     image_height = self.image_height
    #     image_width = self.image_width
    #     image_channels = 1


    #     train_dicom_paths = self.dicom_reader.load_dicom_train_paths()

    #     #if no max, set max to the number of train items
    #     if max_num==None:
    #         max_num = len(train_dicom_paths)
    #     #limit number of training images to max_num
    #     else:
    #         train_dicom_paths = train_dicom_paths[:max_num]



    #     df_full = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


    #     #initialize feature and target images to zeroes
    #     X_train = np.zeros((len(train_dicom_paths), image_height, image_width, image_channels), dtype=np.uint8) #is type 8-bit for pixel intensity values
    #     # Y_train = np.zeros((len(train_dicom_paths), image_height, image_width, 1), dtype=np.bool) #is type bool because if pixel is 1, there is mask, 0 otherwise
    #     Y_train = np.zeros((len(train_dicom_paths)), dtype=np.uint8) #is type bool because if pixel is 1, there is mask, 0 otherwise



    #     print('Getting train images and masks ... ')
    #     sys.stdout.flush() #?


    #     # for n, _id in tqdm_notebook(enumerate(train_dicom_paths), total=len(train_dicom_paths)):

    #     for i, image_path in enumerate(train_dicom_paths):
    #         dicom_image = self.dicom_reader.get_dicom_obj(image_path)

    #         #extracts image_id from the file path
    #         image_id = image_path.split('\\')[-1].replace(".dcm", "")
    #         print("Image id: "+str(image_id))

    #         # print("Shape: "+str(dicom_image.pixel_array.shape))

    #         # #apply dicom pixels
    #         X_train[i] = np.expand_dims(dicom_image.pixel_array, axis=2)

    #         masks = self.data_handler.find_masks(image_id=image_id, dataset=df_full)

    #         print("Num masks: "+str(len(masks)))
    #         # input()
    #         # continue


    #         try:
    #             #if no masks for image, then skip
    #             if len(masks)==0:
    #                 continue
    #             else:
    #                 # if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
    #                 last_mask = masks[-1]

    #                 ## To do:
    #                 ##   Account for all masks found 
    #                 ## 

    #                 #converts to boolean
    #                 # Y_train[i] = np.expand_dims(rle2mask(last_mask, image_height, image_width), axis=2)


    #                 Y_train[i] = 1


    #         except KeyError:
    #             print("Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
    #             # Y_train[n] = np.zeros((image_height, image_width, 1)) # Assume missing masks are empty masks.
    #             Y_train[i] = 0


    #     # #plots the scans and their masks
    #     # for x in range(0, len(X_train)):
    #     #     self.dicom_reader.plot_dicom(X_train[x], Y_train[x])

    #     print("X_train size: "+str(len(X_train)))
    #     print("Y_train size: "+str(len(Y_train)))


    #     return X_train, Y_train


    #returns list of paths to processed image files
    def get_processed_image_paths(self, balanced=False, max_num=None):
        image_paths = self.dicom_reader.load_filtered_dicom_train_paths()


        #if user wants paths where it's 50% positive, and 50% negative
        if balanced:
            print("Loading balanced amount of positive and negative targets")
            new_image_paths = []
            negative = []
            positive = []

            for i in range(0, len(image_paths)):
                image_path = image_paths[i]

                #extracts image_id from the file path
                image_id = image_paths[i].split('\\')[-1].replace("."+str(self.image_preprocessor.preprocessed_ext), "")
                # print("Image id: "+str(image_id))

                masks = self.data_handler.find_masks(image_id=image_id)

                #if non-pneumothorax
                if len(masks)==0:
                    if (max_num==None and len(negative)<int(len(image_paths)/2)) or (max_num!=None and len(negative)<int(max_num/2)):
                        negative.append(image_path)
                        print("Loaded negative signal")
                #if pneumothorax
                else:
                    if (max_num==None and len(positive)<int(len(image_paths)/2)) or (max_num!=None and len(positive)<int(max_num/2)):
                        positive.append(image_path)
                        print("Loaded positive signal")

                #if found enough
                if max_num!=None and len(positive)+len(negative)>=max_num:
                    break

                # print("Num negative: "+str(len(negative))+" | Num positive: "+str(len(positive)))

            #resize negative and positive arrays to match each other
            if len(positive)>len(negative):
                positive = positive[:len(negative)]
            elif len(positive)<len(negative):
                negative = negative[:len(positive)]

            # print("Num positive: "+str(len(positive)))
            # print("Num negative: "+str(len(negative)))

            #adds each side randomly 
            print("Shuffling inputs")
            x = 0
            y = 0
            random.seed(12345)
            while x<len(positive) and y<len(negative):
                #choose positive or negative randomly
                rand_num = random.random()
                if rand_num<0.5:
                    new_image_paths.append(positive[x])
                    x+=1
                    # print("Random positive")
                else:
                    new_image_paths.append(negative[y])
                    y+=1
                    # print("Random negative")

            #if not all the positive symbols were able to be added, add the rest
            if x<len(positive):
                new_image_paths.extend(positive[x:])
            elif y<len(negative):
                new_image_paths.extend(negative[y:])

            # print("Num images: "+str(len(new_image_paths)))

            image_paths = new_image_paths


        #if user doesn't want an explicit balance of positive and negative signals
        else:
            #if no max, set max to the number of train items
            if max_num!=None:
                image_paths = image_paths[:max_num]

        return image_paths




    #just to make sure a series of data isn't related to each other and mistraining the model
    def randomize_data(self):
        pass

    #returns statistical measures related to confusion matrix
    def calculate_statistical_measures(self, confusion_matrix):
        statistical_measures = {}


        try:
            #True Negative
            TN = confusion_matrix[0][0]
            statistical_measures['TN'] = TN
            #False Positive
            FP = confusion_matrix[0][1]
            statistical_measures['FP'] = FP
            #False Negative
            FN = confusion_matrix[1][0]
            statistical_measures['FN'] = FN
            #True Positive
            TP = confusion_matrix[1][1]
            statistical_measures['TP'] = TP
        except Exception as error:
            print("Mishapen confusion matrix") 
            return {}


        #https://en.wikipedia.org/wiki/Confusion_matrix

        if (TN+FP+FN+TP)>0:
            accuracy = (TN+TP)/(TN+FP+FN+TP)
        else:
            accuracy = 0
        statistical_measures['accuracy'] = accuracy

        if TP>0:
            precision = TP/(FP+TP)
        else:
            precision = 0
        statistical_measures['precision'] = precision

        if TN>0:
            negative_predictive_value = TN/(TN+FN)
        else:
            negative_predictive_value = 0
        statistical_measures['NPV'] = negative_predictive_value


        if TP>0:
            F1 = (2*TP) / (2*TP + FP + FN) 
        else:
            F1 = 0
        statistical_measures['F1'] = F1


        if TP > 0:
            sensitivity = TP/(TP+FN)
        else:
            sensitivity = 0
        statistical_measures['sensitivity'] = sensitivity



        #Matthews correlation coefficient (great for binary classification)
        #https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        try:
            MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        except:
            MCC = 0
        statistical_measures['MCC'] = MCC


        #when it's a downmove, how often does model predict a downmove?
        if TN>0:
            specificity = TN/(TN+FP)
        else:
            specificity = 0
        statistical_measures['specificity'] = specificity


        #False Positive Rate (FPR) (Fall-Out)
        FPR = 1 - specificity
        statistical_measures['FPR'] = FPR

        #both TPR (sensitivity) and FPR (100-Specificity) are used to plot the ROC (Receiver Operating Characteristic) curve
        #https://www.medcalc.org/manual/roc-curves.php
        ROC = sensitivity - FPR
        statistical_measures['ROC'] = ROC


        #want to get 200%, 100% for sensitivity and 100% for specificity
        #https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        total = sensitivity + specificity
        statistical_measures['total'] = total


        return statistical_measures


    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)

    #returns CNN
    def create_CNN(self):
        # Initialising the CNN
        classifier = Sequential()

        CNN_size = 32
        pool_size = (3,3)
        filter_size = (3,3)
        CNN_activation = "selu"
        dense_activation = "selu"
        output_activation = "sigmoid"
        # loss = "binary_crossentropy"
        loss = "mean_squared_error"

        classifier.add(Convolution2D(CNN_size, filter_size, input_shape = (self.image_width, self.image_height, 1), padding="same", activation = CNN_activation))

        # Step 2 - Pooling
        #pooling uses a 2x2 or something grid (most of the time is 2x2), goes over the feature maps, and the largest values as its going over become the values in the pooled map
        #slides with a stride of 2. At the end, the pool map should be (length/2)x(width/2)
        classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(BatchNormalization(axis=3))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(BatchNormalization(axis=3))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(BatchNormalization(axis=3))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(BatchNormalization(axis=3))
        classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(Dropout(0.25))

        #flattents the layers
        classifier.add(Flatten())

        #128 is an arbitrary number that can be decreased to lower computation time, and increased for better accuracy
        classifier.add(Dense(units = 128, activation = dense_activation))
        classifier.add(Dropout(0.25))

        classifier.add(Dense(units = 1, activation = output_activation))

        # classifier.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
        classifier.compile(optimizer = 'adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        classifier.summary()

        return classifier


    #returns U-net
    #architecture source: https://github.com/yihui-he/u-net
    def create_Unet(self):

        start_size = 16
        pool_size = (2,2)
        filter_size = (3,3)
        conv_activation = "selu"
        dense_activation = "selu"
        output_activation = "sigmoid"
        # loss = "mean_squared_error"

        # inputs = Input((self.image_width, self.image_height, 1))
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)



        # # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # # convdeep = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
        # # convdeep = Conv2D(1024, (3, 3), activation='relu', padding='same')(convdeep)
        
        # # upmid = concatenate([Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], axis=1)
        # # convmid = Conv2D(512, (3, 3), activation='relu', padding='same')(upmid)
        # # convmid = Conv2D(512, (3, 3), activation='relu', padding='same')(convmid)




        # up6 = concatenate([Conv2D(256, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], axis=1)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=1)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        # up8 = concatenate([Conv2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], axis=1)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        # up9 = concatenate([Conv2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], axis=1)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        # conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
        # # model = Model(input=inputs, output=conv10)






        # inputs = Input((self.image_width, self.image_height, 1))
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        # # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        # # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)



        # # up6 = concatenate([Conv2D(256, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], axis=1)
        # # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        # # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=1)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        # up8 = concatenate([Conv2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], axis=1)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        # up9 = concatenate([Conv2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], axis=1)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        # conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
        # # model = Model(input=inputs, output=conv10)





        inputs = Input((self.image_width, self.image_height, 1))
        conv1 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(inputs)
        conv1 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(pool1)
        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(pool2)
        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=1)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2D(start_size*2, pool_size, activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv3)), conv2], axis=1)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(up8)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv8)

        up9 = concatenate([Conv2D(start_size*1, pool_size,activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv8)), conv1], axis=1)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(up9)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv9)

        conv10 = Convolution2D(1, (1, 1), activation=output_activation)(conv9)
        # model = Model(input=inputs, output=conv10)









        dense1 = Flatten()(conv10)

        dense2 = Dense(units = 1, activation = output_activation)(dense1)
        # # # classifier.add(Dropout(0.25))

        
        model = Model(input=inputs, output=dense2)

        model.compile(optimizer='adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        model.summary()

        return model


    #trains CNN
    def train(self):

        max_images = 30
        X = self.get_processed_image_paths(balanced=True, max_num=max_images)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


        # return




        train_ratio = 0.7
        validation_ratio = 0.2
        X_train, X_validate, X_test = self.data_handler.split_data(X, train_ratio, validation_ratio)
        # Y_train, Y_validate, Y_test = self.data_handler.split_data(Y, train_ratio, validation_ratio)

        print("X_train: "+str(len(X_train)))
        print("X_validate: "+str(len(X_validate)))

        # print("X_train: "+str(X_train.shape))
        # print("X_validate: "+str(X_validate.shape))
        # print("X_test: "+str(X_test.shape))
        # print()
        # print("Y_train: "+str(Y_train.shape))
        # print("Y_validate: "+str(Y_validate.shape))
        # print("Y_test: "+str(Y_test.shape))
        # print()
        # print("Num pneumothorax in train: "+str(np.sum(Y_train))+"/"+str(Y_train.shape[0]))
        # print("Num pneumothorax in validate: "+str(np.sum(Y_validate))+"/"+str(Y_validate.shape[0]))
        # print()



        # X_train_normalized, X_validate_normalized, X_test_normalized, X_normalization_params = self.image_preprocessor.normalize_data(X_train, X_validate, X_test)



        # print("X_train: "+str(X_train_normalized.shape))
        # print("X_validate: "+str(X_validate_normalized.shape))
        # print("X_test: "+str(X_test_normalized.shape))
        # print()
        # print("Y_train: "+str(len(Y_train_normalized)))
        # print("Y_validate: "+str(len(Y_validate_normalized)))
        # print("Y_test: "+str(len(Y_test_normalized)))
        # print()


        # classifier = self.create_CNN()
        classifier = self.create_Unet()



        # # construct the image generator for data augmentation
        # datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        #     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        #     horizontal_flip=True, fill_mode="nearest")

        # datagen.fit(X_train_normalized)

        
        


        # Parameters
        params = {'dim': (self.image_height,self.image_width,1),
                  'shuffle': True}

        batch_size = 10
        training_generator = DataGenerator(X_train, Y, batch_size, **params)



        #trains
        epochs = 5

        # #normal training
        # classifier.fit(X_train_normalized, Y_train,
        #                batch_size=batch_size,
        #                epochs=epochs)

        # # fits the model on batches with real-time data augmentation:
        # classifier.fit_generator(datagen.flow(X_train_normalized, Y_train, batch_size=batch_size),
        #                         steps_per_epoch=len(X_train_normalized) / batch_size,
        #                         epochs=epochs)

        # fits the model on batches with real-time data augmentation:
        classifier.fit_generator(generator=training_generator,
                                steps_per_epoch=len(X_train) / batch_size,
                                epochs=epochs)




        # # here's a more "manual" example
        # for e in range(epochs):
        #     print('Epoch', e)

        #     batches = 0
        #     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        #         model.fit(x_batch, y_batch)
        #         batches += 1
        #         if batches >= len(x_train) / 32:
        #             # we need to break the loop by hand because
        #             # the generator loops indefinitely
        #             break




        ## Plot training history ##
        # # plot the training loss and accuracy
        # plt.style.use("ggplot")
        # plt.figure()
        # N = EPOCHS
        # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        # plt.title("Training Loss and Accuracy on Santa/Not Santa")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend(loc="lower left")
        # plt.savefig(args["plot"])



        from sklearn.metrics import confusion_matrix

        batch_size = 10
        validation_generator = DataGenerator(X_validate, Y, batch_size, **params)

        #predicts
        preds = classifier.predict_generator(validation_generator)


        print("predictions: "+str(preds.shape))


        #gets actual validation labels
        X_validate_images, y_validate_non_category = validation_generator.get_processed_images(start=0, end=len(X_validate))
        #gets predicted validation labels
        y_predict_non_category = [ t>0.5 for t in preds]

        print("validation: "+str(y_validate_non_category.shape))
        print("Validation labels: "+str(y_validate_non_category[:10]))
        print("Predicted labels: "+str(preds[:10]))
        # print(y_predict_non_category)

        #calculates confusion matrix
        conf_matrix = confusion_matrix(y_validate_non_category, y_predict_non_category)
        print("Confusion matrix: "+str(conf_matrix))

        print()
        statistical_measures = self.calculate_statistical_measures(conf_matrix)

        for measure in statistical_measures:
            print(str(measure)+": "+str(statistical_measures[measure]))




        #to save a model: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        classifier.save('./trained_models/cnn_model.h5')





if __name__=="__main__":
    CNN_classifier = CNNClassifier()

    CNN_classifier.train()
    
    # CNN_classifier.get_unprocessed_feature_target_images()