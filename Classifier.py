from DICOM_reader import DICOMReader
from DataHandler import DataHandler
from ImagePreprocessor import *
from DataGenerator import DataGenerator

from mask_functions import rle2mask

import sys
import os
import random

#ML libraries
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import keras
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

from keras.layers import *
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import imageio

#For my windows machine
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


"""

Handles differen types (binary, segmentation) classification training, validation, and testing

"""
class Classifier(ABC):

    image_width = 512
    image_height = 512

    model_archs = {"cnn": "cnn", "unet": "unet"}

    dicom_reader = None
    data_handler = None
    image_preprocessor = None

    def __init__(self):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()
        self.image_preprocessor = ChestRadiograph()

    #returns proper filename prefix for specified model_arch, e.x: "cnn" returns "cnn" for use in "./cnn_model.h5"
    def get_model_arch_filename_prefix(self, model_arch):
        try:
            return self.model_archs[model_arch.lower()]
        except Exception as error:
            return ""

    #returns list of paths to processed image files
    #dataset_type = {"train", "test"}
    #label_type = {"binary", "segmentation"}
    def get_processed_image_paths(self, dataset_type="train", label_type="binary", balanced=False, max_num=None):

        if dataset_type.lower() == "train":
            image_paths = self.dicom_reader.load_filtered_dicom_train_paths()
        elif dataset_type.lower() == "test":
            image_paths = self.dicom_reader.load_filtered_dicom_test_paths()


        #if user wants paths where it's 50% positive, and 50% negative
        if balanced:
            print("Loading balanced amount of positive and negative targets")
            new_image_paths = []
            negative = []
            positive = []

            for i in range(0, len(image_paths)):
                image_path = image_paths[i]

                #extracts image_id from the file path
                image_id = self.dicom_reader.extract_image_id(image_paths[i], self.image_preprocessor.preprocessed_ext)

                #finds positive mask associated with the image
                masks = self.data_handler.find_masks(image_id=image_id)

                #if Negative
                if len(masks)==0:
                    #makes sure haven't added too many negative labels
                    if (max_num==None and len(negative)<int(len(image_paths)/2)) or (max_num!=None and len(negative)<int(max_num/2)):
                        negative.append(image_path)
                #if Positive
                else:
                    #makes sure haven't added too many positive labels
                    if (max_num==None and len(positive)<int(len(image_paths)/2)) or (max_num!=None and len(positive)<int(max_num/2)):
                        positive.append(image_path)

                #if found enough of both positive and negative
                if max_num!=None and len(positive)+len(negative)>=max_num:
                    break



            #resize negative and positive arrays to match each other
            if len(positive)>len(negative):
                positive = positive[:len(negative)]
            elif len(positive)<len(negative):
                negative = negative[:len(positive)]


            # #adds each side randomly 
            # print("Shuffling inputs")
            # x = 0
            # y = 0
            # random.seed(12345)
            # while x<len(positive) and y<len(negative):
            #     #choose positive or negative randomly
            #     rand_num = random.random()
            #     if rand_num<0.5:
            #         new_image_paths.append(positive[x])
            #         x+=1
            #     else:
            #         new_image_paths.append(negative[y])
            #         y+=1

            # #if not all the positive symbols were able to be added, add the rest
            # if x<len(positive):
            #     new_image_paths.extend(positive[x:])
            # elif y<len(negative):
            #     new_image_paths.extend(negative[y:])

            # image_paths = new_image_paths



            #shuffles positives and negatives
            image_paths = []
            image_paths.extend(positive)
            image_paths.extend(negative)
            random.seed(12345)
            random.shuffle(image_paths)


        #if user doesn't want an explicit balance of positive and negative signals
        else:
            #if no max, set max to the number of train items
            if max_num!=None and max_num<=len(image_paths) and max_num>0:
                image_paths = image_paths[:max_num]


        return image_paths




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
        except Exception as error:
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

        #False Negative Rate (FNR)
        FNR = 1 - sensitivity
        statistical_measures['FNR'] = FNR

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
        print("Y_true: "+str(y_true.shape))
        print("Y_pred: "+str(y_pred.shape))

        return 1-self.dice_coef(y_true, y_pred)

    #returns CNN
    @abstractmethod
    def create_CNN(self):
        # Initialising the CNN
        classifier = Sequential()

        CNN_size = 16
        pool_size = (3,3)
        filter_size = (3,3)
        CNN_activation = "selu"
        dense_activation = "selu"
        output_activation = "linear"
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

        #flattents the layers
        classifier.add(Flatten())

        #128 is an arbitrary number that can be decreased to lower computation time, and increased for better accuracy
        classifier.add(Dense(units = 128, activation = dense_activation))
        classifier.add(Dropout(0.25))

        classifier.add(Dense(units = 1, activation = output_activation))

        classifier.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
        # classifier.compile(optimizer = 'adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        classifier.summary()

        return classifier


    #returns U-net
    #architecture source: https://github.com/yihui-he/u-net
    # and https://github.com/jocicmarko/ultrasound-nerve-segmentation/
    @abstractmethod
    def create_Unet(self):

        start_size = 16
        pool_size = (2,2)
        filter_size = (3,3)
        conv_activation = "selu"
        dense_activation = "selu"
        output_activation = "sigmoid"
        # loss = "mean_squared_error"


        inputs = Input((self.image_width, self.image_height, 1))
        conv1 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(inputs)
        conv1 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(pool1)
        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(pool2)
        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=3)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2D(start_size*2, pool_size, activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv3)), conv2], axis=3)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(up8)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2D(start_size*1, pool_size,activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv8)), conv1], axis=3)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(up9)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        # conv10 = Convolution2D(1, (1, 1), activation=output_activation)(conv9)

        conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)
        # model = Model(inputs=inputs, outputs=conv10)





        dense1 = Flatten()(conv10)

        dense2 = Dense(units = 1, activation = output_activation)(dense1)
        # # # # classifier.add(Dropout(0.25))

        
        model = Model(inputs=inputs, outputs=dense2)

        # model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        model.compile(optimizer=Adam(lr=1e-5), loss="mean_squared_error", metrics=["accuracy"])

        model.summary()

        return model

    #returns parameters for the data generator depending on what procress of teh model creation we are in
    def get_data_generator_params(self, step="train"):
        # Parameters
        params = {'dim': (self.image_height,self.image_width,1),
                  'augment': True,
                  'shuffle': True}

        if step.lower() == "train":
            params['augment'] = True
            params['shuffle'] = True

        elif step.lower() == "test":
            params['augment'] = False
            params['shuffle'] = False

        return params

    #returns Data Generator
    @abstractmethod
    def create_data_generator(self, feature_dataset, label_dataset, batch_size, step="train"):
        params = self.get_data_generator_params(step=step)

        data_generator = DataGenerator(feature_dataset, label_dataset, batch_size, "binary", **params)

        return data_generator


    #predicts on the dataset, and prints a confusion matrix of the results
    def predict(self, classifier, feature_dataset, full_label_dataset, batch_size=1):

        # #Validation prediction
        # params = {'dim': (self.image_height,self.image_width,1),
        #           'augment': False,
        #           'shuffle': False}

        # generator = DataGenerator(feature_dataset, full_label_dataset, batch_size, **params)

        generator = self.create_data_generator(feature_dataset, full_label_dataset, batch_size, "test")

        print(generator)

        preds = classifier.predict_generator(generator)


        print("predictions: "+str(preds.shape))


        #gets actual labels
        X_images, y_non_category = generator.get_processed_images(start=0, end=len(feature_dataset))
        #gets predicted labels
        y_predict_non_category = [ t>0.5 for t in preds]

        #prints number of example labels and their predictions
        print()
        print("Example (First 10): ")
        for x in range(0, 10):
            print("  Actual: "+str(y_non_category[x])+", Pred: "+str(preds[x]))
        print()


        #calculates confusion matrix
        conf_matrix = confusion_matrix(y_non_category, y_predict_non_category)
        print("Confusion matrix: ")
        print(conf_matrix)

        
        stats = self.calculate_statistical_measures(conf_matrix)

        print()
        print("Accuracy: "+str(stats['accuracy']))
        print()
        print("False Positive Rate: "+str(stats['FPR']))
        print("False Negative Rate: "+str(stats['FNR']))
        print("PPV (Precision): "+str(stats['precision']))
        print()
        print("Specificity: "+str(stats['specificity']))
        print("Sensitivity: "+str(stats['sensitivity']))
        print("Total (specificity + sensitivity): "+str(stats['total']))
        print()
        print("ROC: "+str(stats['ROC']))
        print("F1: "+str(stats['F1']))
        print("MCC: "+str(stats['MCC']))


    #trains CNN
    def train(self, model_arch="cnn", dataset_size=100):

        max_images = dataset_size
        X = self.get_processed_image_paths(balanced=True, max_num=max_images)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


        train_ratio = 0.7
        validation_ratio = 0.2
        X_train, X_validate, X_test = self.data_handler.split_data(X, train_ratio, validation_ratio)

        print("X_train size: "+str(len(X_train)))
        print("X_validate size: "+str(len(X_validate)))


        if model_arch.lower() == "cnn": 
            classifier = self.create_CNN()  
        elif model_arch.lower() == "unet":
            classifier = self.create_Unet()
        else:
            print("Unsupported specified model type")
            return

        # params = self.get_data_generator_params("train")

        batch_size = 10
        epochs = 10

        # training_generator = DataGenerator(X_train, Y, batch_size, **params)

        training_generator = self.create_data_generator(X_train, Y, batch_size, "train")

        # fits the model on batches with real-time data augmentation:
        classifier.fit_generator(generator=training_generator,
                                steps_per_epoch=len(X_train) / batch_size,
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


        #perform prediction on validation portion
        self.predict(feature_dataset=X_validate, full_label_dataset=Y, batch_size=10)


        classifier.save("./trained_models/"+str(self.get_model_arch_filename_prefix(model_arch))+"_model.h5")



    #performs predictions and subsequent statistic calculations on unofficial test dataset
    def test(self, model_arch="cnn", dataset_size=100):
        max_images = dataset_size
        X = self.get_processed_image_paths(dataset_type="test", balanced=False, max_num=max_images)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths



        #loads the model
        try:
            classifier = load_model("./trained_models/"+str(self.get_model_arch_filename_prefix(model_arch))+"_model.h5")
        except Exception as error:
            print(error)
            print("Model doesn't exist for "+str(model_arch))
            return



        self.predict(classifier=classifier, feature_dataset=X, full_label_dataset=Y, batch_size=10)





"""

Handles binary classification training, validation, and testing

"""
class BinaryClassifier(Classifier):
    def __init__(self):
        super().__init__()
        pass

    def print_something(self):
        print("Something")

    #creates CNN sculpted for binary classification
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
        optimizer = "adam"

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

        classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
        # classifier.compile(optimizer = 'adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        classifier.summary()

        return classifier


    #creates and returns U-net architecture model
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
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(pool1)
        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(pool2)
        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=3)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2D(start_size*2, pool_size, activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv3)), conv2], axis=3)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(up8)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2D(start_size*1, pool_size,activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv8)), conv1], axis=3)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(up9)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        # conv10 = Convolution2D(1, (1, 1), activation=output_activation)(conv9)

        conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)
        # model = Model(inputs=inputs, outputs=conv10)


        dense1 = Flatten()(conv10)

        dense2 = Dense(units = 1, activation = output_activation)(dense1)
        # # # # classifier.add(Dropout(0.25))

        
        model = Model(inputs=inputs, outputs=dense2)

        # model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        model.compile(optimizer=Adam(lr=1e-5), loss="mean_squared_error", metrics=["accuracy"])

        model.summary()

        return model

    #creates custom data generator specific for labels of type binary
    def create_data_generator(self, feature_dataset, label_dataset, batch_size, step="train"):
        params = self.get_data_generator_params(step=step)

        generator = DataGenerator(feature_dataset, label_dataset, batch_size, "binary", **params)
        return generator


"""

Handles segmentation classification training, validation, and testing

"""
class SegmentationClassifier(Classifier):
    def __init__(self):
        super().__init__()
        pass

    #creates CNN sculpted for segmentation classification
    def create_CNN(self):
        # Initialising the CNN
        classifier = Sequential()

        CNN_size = 32
        pool_size = (3,3)
        filter_size = (3,3)
        conv_activation = "selu"
        dense_activation = "selu"
        output_activation = "sigmoid"
        # loss = "binary_crossentropy"
        loss = "mean_squared_error"
        optimizer = "adam"

        # classifier.add(Convolution2D(CNN_size, filter_size, input_shape = (self.image_width, self.image_height, 1), padding="same", activation = CNN_activation))

        # # Step 2 - Pooling
        # #pooling uses a 2x2 or something grid (most of the time is 2x2), goes over the feature maps, and the largest values as its going over become the values in the pooled map
        # #slides with a stride of 2. At the end, the pool map should be (length/2)x(width/2)
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(0.25))

        # # # Adding a second convolutional layer
        # # classifier.add(Convolution2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
        # # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # # classifier.add(BatchNormalization(axis=3))
        # # classifier.add(Dropout(0.25))

        # # # Adding a second convolutional layer
        # # classifier.add(Convolution2D(CNN_size, filter_size, activation = CNN_activation))
        # # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(Dropout(0.25))

        # # Adding a second convolutional layer
        # classifier.add(Convolution2D(1, (1,1), activation = output_activation))
        # # classifier.add(MaxPooling2D(pool_size = pool_size))
        # # classifier.add(Dropout(0.25))

        # classifier.compile(optimizer = 'adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        # classifier.summary()

        # return classifier




        inputs = Input((self.image_width, self.image_height, 1))
        conv1 = Conv2D(CNN_size , filter_size, activation=conv_activation, padding='same')(inputs)
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)
        dropout1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout1)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)
        dropout2 = Dropout(0.25)(pool2)

        conv3 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout2)
        pool3 = MaxPooling2D(pool_size=pool_size)(conv3)
        dropout3 = Dropout(0.25)(pool3)

        conv4 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout3)
        pool4 = MaxPooling2D(pool_size=pool_size)(conv4)
        dropout4 = Dropout(0.25)(pool4)

        conv10 = Conv2D(1, (1, 1), activation=output_activation)(dropout4)
        # conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)


        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])

        model.summary()

        return model


    #creates and returns U-net architecture model specific for segmentation prediction
    def create_Unet(self):
        start_size = 16
        pool_size = (2,2)
        filter_size = (3,3)
        conv_activation = "selu"
        output_activation = "sigmoid"


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
        # conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(pool1)
        conv2 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv2)
        # conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(pool2)
        conv3 = Conv2D(start_size*4, filter_size, activation=conv_activation, padding='same')(conv3)
        # conv3 = BatchNormalization()(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        # up7 = concatenate([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=3)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2D(start_size*2, pool_size, activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv3)), conv2], axis=3)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(up8)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv8)
        # conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2D(start_size*1, pool_size,activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv8)), conv1], axis=3)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(up9)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv9)
        # conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, (1, 1), activation=output_activation)(conv9)
        # conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)


        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])

        model.summary()

        return model


    #creates custom data generator specific for labels of type segments
    def create_data_generator(self, feature_dataset, label_dataset, batch_size, step="train"):
        params = self.get_data_generator_params(step=step)

        #can't easily augment, because that'll change mask/label positions
        params['augment'] = False

        generator = DataGenerator(feature_dataset, label_dataset, batch_size, "segment", **params)
        return generator





















if __name__=="__main__":
    classifier = SegmentationClassifier()

    # CNN_classifier.train(dataset_size=200)

    classifier.train(model_arch="unet", dataset_size=30)
    