from DICOM_reader import DICOMReader
from DataHandler import DataHandler
from ImagePreprocessor import ImagePreprocessor

from mask_functions import rle2mask

import sys
import os

#ML libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

#For my windows machine
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


"""

Handles CNN training, validation, and testing

"""
class CNNClassifier:

    image_width = 1024
    image_height = 1024

    dicom_reader = None
    data_handler = None
    image_preprocessor = None

    def __init__(self):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()
        self.image_preprocessor = ImagePreprocessor()

    #returns 1024x1024 raw pixel data of feature dicom images
    # feature images will have pixel intensities
    # target images will have binary denoting masks of pneumothorax
    def get_unprocessed_feature_target_images(self, max_num=None):
        image_height = self.image_height
        image_width = self.image_width
        image_channels = 1


        train_dicom_paths = self.dicom_reader.load_dicom_train_paths()

        #if no max, set max to the number of train items
        if max_num==None:
            max_num = len(train_dicom_paths)
        #limit number of training images to max_num
        else:
            train_dicom_paths = train_dicom_paths[:max_num]



        df_full = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


        #initialize feature and target images to zeroes
        X_train = np.zeros((len(train_dicom_paths), image_height, image_width, image_channels), dtype=np.uint8) #is type 8-bit for pixel intensity values
        # Y_train = np.zeros((len(train_dicom_paths), image_height, image_width, 1), dtype=np.bool) #is type bool because if pixel is 1, there is mask, 0 otherwise
        Y_train = np.zeros((len(train_dicom_paths)), dtype=np.uint8) #is type bool because if pixel is 1, there is mask, 0 otherwise



        print('Getting train images and masks ... ')
        sys.stdout.flush() #?


        # for n, _id in tqdm_notebook(enumerate(train_dicom_paths), total=len(train_dicom_paths)):

        for i, image_path in enumerate(train_dicom_paths):
            dicom_image = self.dicom_reader.get_dicom_obj(image_path)

            #extracts image_id from the file path
            image_id = image_path.split('\\')[-1].replace(".dcm", "")
            print("Image id: "+str(image_id))

            # print("Shape: "+str(dicom_image.pixel_array.shape))

            # #apply dicom pixels
            X_train[i] = np.expand_dims(dicom_image.pixel_array, axis=2)

            masks = self.data_handler.find_masks(image_id=image_id, dataset=df_full)

            print("Num masks: "+str(len(masks)))
            # input()
            # continue


            try:
                #if no masks for image, then skip
                if len(masks)==0:
                    continue
                else:
                    # if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
                    last_mask = masks[-1]

                    ## To do:
                    ##   Account for all masks found 
                    ## 

                    #converts to boolean
                    # Y_train[i] = np.expand_dims(rle2mask(last_mask, image_height, image_width), axis=2)


                    Y_train[i] = 1


            except KeyError:
                print("Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
                # Y_train[n] = np.zeros((image_height, image_width, 1)) # Assume missing masks are empty masks.
                Y_train[i] = 0


        # #plots the scans and their masks
        # for x in range(0, len(X_train)):
        #     self.dicom_reader.plot_dicom(X_train[x], Y_train[x])

        print("X_train size: "+str(len(X_train)))
        print("Y_train size: "+str(len(Y_train)))


        return X_train, Y_train

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
            return statistical_measures


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

    #returns CNN
    def create_CNN(self):
        # Initialising the CNN
        classifier = Sequential()

        CNN_size = 16
        pool_size = (3,3)
        CNN_activation = "relu"
        dense_activation = "relu"
        output_activation = "sigmoid"

        classifier.add(Convolution2D(CNN_size, (3, 3), input_shape = (self.image_width, self.image_height, 1), activation = CNN_activation))

        # Step 2 - Pooling
        #pooling uses a 2x2 or something grid (most of the time is 2x2), goes over the feature maps, and the largest values as its going over become the values in the pooled map
        #slides with a stride of 2. At the end, the pool map should be (length/2)x(width/2)
        classifier.add(MaxPooling2D(pool_size = pool_size))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, (3, 3), activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, (3, 3), activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(CNN_size, (3, 3), activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, (3, 3), activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, (3, 3), activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(Dropout(0.25))

        #flattents the layers
        classifier.add(Flatten())

        #128 is an arbitrary number that can be decreased to lower computation time, and increased for better accuracy
        classifier.add(Dense(units = 128, activation = dense_activation))
        classifier.add(Dropout(0.25))

        classifier.add(Dense(units = 1, activation = output_activation))

        classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

        classifier.summary()

        return classifier



    #trains CNN
    def train(self):

        # #labels are dataframe where keys are column names, and values are rows of that column
        # labels = self.data_handler.read_train_labels()

        # #converts DataFrame to 2D list
        # labels = labels.values.tolist()

        # print("Num labels: "+str(len(labels)))


        max_images = 5
        X, Y = self.get_unprocessed_feature_target_images(max_images)

        dcm_image = X[0]

        #reshapes for display
        dcm_image = np.squeeze(dcm_image, axis=2)
        # self.dicom_reader.plot_pixel_array(dcm_image)

        # dcm_image = self.image_preprocessor.apply_gaussian_blur(dcm_image)
        dcm_image = self.image_preprocessor.canny_edge_detector(dcm_image)
        # dcm_image = np.squeeze(dcm_image, axis=2)
        self.dicom_reader.plot_pixel_array(dcm_image)


        return





        train_ratio = 0.7
        validation_ratio = 0.2
        X_train, X_validate, X_test = self.data_handler.split_data(X, train_ratio, validation_ratio)
        Y_train, Y_validate, Y_test = self.data_handler.split_data(Y, train_ratio, validation_ratio)
        # for label in labels:
        #   print(label)

        print("X_train: "+str(X_train.shape))
        print("X_validate: "+str(X_validate.shape))
        print("X_test: "+str(X_test.shape))
        print()
        print("Y_train: "+str(Y_train.shape))
        print("Y_validate: "+str(Y_validate.shape))
        print("Y_test: "+str(Y_test.shape))
        print()



        X_train_normalized, X_validate_normalized, X_test_normalized, X_normalization_params = self.image_preprocessor.normalize_data(X_train, X_validate, X_test)

        #Y is already normalized
        # Y_train_normalized, Y_validate_normalized, Y_test_normalized, Y_normalization_params = self.normalize_data(Y_train, Y_validate, Y_test)


        print("X_train: "+str(X_train_normalized.shape))
        print("X_validate: "+str(X_validate_normalized.shape))
        print("X_test: "+str(X_test_normalized.shape))
        print()
        # print("Y_train: "+str(len(Y_train_normalized)))
        # print("Y_validate: "+str(len(Y_validate_normalized)))
        # print("Y_test: "+str(len(Y_test_normalized)))
        # print()

        # print("X_train: "+str(X_train[0]))

        # print("X_train_normalized: "+str(X_train_normalized[0]))


        classifier = self.create_CNN()


        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")

        X_train_new_images = aug.flow(X_train_normalized, Y_train, batch_size=32)


        # Y_train = keras.utils.to_categorical(Y_train, 1)


        
        print("X_train: "+str(X_train_normalized.shape))
        print("X_validate: "+str(X_validate_normalized.shape))
        print("Y_train: "+str(Y_train.shape))
        # print("Len x: "+str(X_train_new_images.shape))


        # # fits the model on batches with real-time data augmentation:
        # classifier.fit_generator(X_train_new_images, steps_per_epoch=len(X_train_new_images) / 32, epochs=epochs)


        ### Should really perform data augmentation ###


        #trains
        batch_size = 10
        epochs = 3
        classifier.fit(X_train_normalized, Y_train,
                       batch_size=batch_size,
                       epochs=epochs)


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

        #predicts
        preds = classifier.predict(X_validate_normalized)




        #gets dimensions of confusion matrix
        # y_validate_non_category = [ np.argmax(t) for t in Y_validate ]
        y_validate_non_category = Y_validate
        y_predict_non_category = [ t>0.5 for t in preds]

        print(y_validate_non_category[:10])
        print(preds[:10])
        # print(y_predict_non_category)

        #calculates confusion matrix
        conf_matrix = confusion_matrix(y_validate_non_category, y_predict_non_category)
        print(conf_matrix)

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