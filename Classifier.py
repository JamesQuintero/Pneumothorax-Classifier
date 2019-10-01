from DICOM_reader import DICOMReader
from DataHandler import DataHandler
from ImagePreprocessor import *
from DataGenerator import DataGenerator

from mask_functions import rle2mask

import sys
import os
import random
import json

#ML libraries
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import keras
from keras import backend as K
from keras.layers import Input
from keras.layers import *

from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

from keras.callbacks import EarlyStopping
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
    project = ""

    image_width = 512
    image_height = 512

    model_archs = {"cnn": "cnn", "unet": "unet"}

    dicom_reader = None
    data_handler = None
    image_preprocessor = None

    hyperparameters = {}

    def __init__(self, project):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()

        self.project = project.lower()
        if self.project=="chest_radiograph":
            self.image_preprocessor = ChestRadiograph()

        self.hyperparameters = self.data_handler.load_hyperparameters()

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


    #returns very simple U-net
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

        up8 = concatenate([Conv2D(start_size*2, pool_size, activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv3)), conv2], axis=3)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(up8)
        conv8 = Conv2D(start_size*2, filter_size, activation=conv_activation, padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2D(start_size*1, pool_size,activation=conv_activation, padding='same')(UpSampling2D(size=pool_size)(conv8)), conv1], axis=3)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(up9)
        conv9 = Conv2D(start_size*1, filter_size, activation=conv_activation, padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)
        # model = Model(inputs=inputs, outputs=conv10)





        dense1 = Flatten()(conv10)

        dense2 = Dense(units = 1, activation = output_activation)(dense1)

        
        model = Model(inputs=inputs, outputs=dense2)

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


    #returns list of individual confusion matrices if segmentation, and list of size 1 for binary
    @abstractmethod
    def calculate_confusion_matrices(self, target_data, prediction_data):

        #calculates confusion matrix
        conf_matrix = confusion_matrix(y_non_category, y_predict_non_category)

        return conf_matrix

    def print_statistical_measures(self, stats):
        print()
        print("----------------------------------")
        print("Accuracy:                "+str(stats['accuracy']))
        print("----------------------------------")
        print("False Positive Rate:     "+str(stats['FPR']))
        print("False Negative Rate:     "+str(stats['FNR']))
        print("PPV (Precision):         "+str(stats['precision']))
        print("----------------------------------")
        print("Specificity:             "+str(stats['specificity']))
        print("Sensitivity:             "+str(stats['sensitivity']))
        print("Total (spec + sens):     "+str(stats['total']))
        print("----------------------------------")
        print("ROC:                     "+str(stats['ROC']))
        print("F1:                      "+str(stats['F1']))
        print("MCC:                     "+str(stats['MCC']))
        print("----------------------------------")
        print()


    def statistical_analysis(self, target_data, prediction_data, verbose=False):

        if verbose:
            print("Target data: "+str(len(target_data)))
            print("Prediction data: "+str(len(prediction_data)))


        #calculates confusion matrix
        conf_matrices = self.calculate_confusion_matrices(target_data, prediction_data)


        agg_conf_matrices = np.zeros(conf_matrices[0].shape)
        all_stats = []
        for x in range(0, len(conf_matrices)):
            conf_matrix = conf_matrices[x]
            
            stats = self.calculate_statistical_measures(conf_matrix)

            if verbose:
                print("Confusion matrix "+str(x)+": ")
                print(conf_matrix)
                self.print_statistical_measures(stats)

            agg_conf_matrices += conf_matrix
            all_stats.append(stats)






        agg_stats = self.calculate_statistical_measures(agg_conf_matrices)

        if verbose:
            print("Aggregate confusion matrix: ")
            print(agg_conf_matrices)
            self.print_statistical_measures(agg_stats)


        average_stats = {}
        #iterates through all statistical metrics
        for stat in all_stats[0]:
            average_stats[stat] = 0

            #iterates through all confusion matrices' states
            for x in range(0, len(all_stats)):
                try:
                    average_stats[stat] += all_stats[x][stat]
                except Exception as error:
                    continue

            average_stats[stat] /= len(all_stats)

        if verbose:
            print()
            print()
            print("Average stats: ")
            self.print_statistical_measures(average_stats)



        return (all_stats, agg_stats)





    #predicts on the dataset, and prints a confusion matrix of the results
    def prediction_analysis(self, classifier, feature_dataset, full_label_dataset, batch_size=1, verbose=False):

        generator = self.create_data_generator(feature_dataset, full_label_dataset, 1, "test")
        preds = classifier.predict_generator(generator)


        print("predictions: "+str(preds.shape))


        #gets actual labels
        X_images, y_non_category = generator.get_processed_images(start=0, end=len(feature_dataset))
        #gets predicted labels
        y_predict_non_category = [ t>0.5 for t in preds]



        return self.statistical_analysis(y_non_category, y_predict_non_category, verbose)


    #Bagging
    def bagging(self):
        pass

    """
    takes a dataset, splits into n_splits+1 sections, then split each of those into train and test sections, 
    then trains a model on each section and ensembles them together to hopefully have one better model that's more stable
    than each model together
    https://www.sciencedirect.com/science/article/pii/S0925231214007644

    Used as an ensemble since a bunch of smaller models might be able to predict better than a large model. 

    """
    # *args are the important arguments for calling train
    def resampling_ensemble(self, n_splits = 0, **train_args):

        print("Train arguments: "+str(train_args))
        dataset_size = train_args['dataset_size']
        X = self.get_processed_image_paths(balanced=train_args['balanced'], max_num=dataset_size)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths

        print("Dataset size: "+str(dataset_size))
        print("n_splits: "+str(n_splits))

        section_size = int(dataset_size/(n_splits+1))

        print("Section size: "+str(section_size))
        print()


        classifiers = []
        X_trains = []
        X_validates = []
        X_tests = []
        Ys = []
        results = []
        for section in range(0, n_splits+1):
            print("Section: "+str(section+1)+"/"+str(n_splits+1))
            start = section*section_size
            end = (section+1)*section_size

            X_section = X[start:end]

            #trains on this section
            train_args['X'] = X_section
            train_args['Y'] = Y


            classifier, X_train_section, X_validate_section, X_test_section, Y_section = self.train(**train_args)
            classifiers.append(classifier)
            X_trains.append(X_train_section)
            X_validates.append(X_validate_section)
            X_tests.append(X_test_section)
            Ys.append(Y_section)

            #performs prediction analysis on results
            prediction_analysis = self.prediction_analysis(classifier=classifier, 
                                                        feature_dataset=X_validate_section, 
                                                        full_label_dataset=Y_section, 
                                                        batch_size=train_args['batch_size'])

            results.append(prediction_analysis)


        print("")


        return classifiers, X_trains, X_validates, X_tests, Ys



    """
    takes a dataset, gets k_folds so that 0.8 of the dataset is for training, 0.2 is for testing, and this portion moves 
    throughout the dataset so that by the end, each portion of the dataset got a chance to be the test portion. 
    https://en.wikipedia.org/wiki/Cross-validation_(statistics)

    Used to judge performance of a model arch on a dataset

    """
    def kfold_cross_validation(self, k_folds = 0, **train_args):

        print("Train arguments: "+str(train_args))
        dataset_size = train_args['dataset_size']
        X = self.get_processed_image_paths(balanced=train_args['balanced'], max_num=dataset_size)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


        seed = 12345

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        stats = []
        for train, test in kfold.split(X, Y):
          # # create model
          #   model = Sequential()
          #   model.add(Dense(12, input_dim=8, activation='relu'))
          #   model.add(Dense(8, activation='relu'))
          #   model.add(Dense(1, activation='sigmoid'))
          #   # Compile model
          #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
          #   # Fit the model
          #   model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)



            # # evaluate the model
            # scores = model.evaluate(X[test], Y[test], verbose=0)
            # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            # stats.append(scores[1] * 100)



            pass




    """
    trains model with model_arch architecture
    returns the Keras model, training dataset, validation dataset, testing dataset, and all labels
    """
    # @abstractmethod
    def train(self, model_arch="cnn", dataset_size=100, balanced=False, X=None, Y=None, train_ratio=0.7, val_ratio=0.2, batch_size=1, epochs=1):

        #load features and targets if they aren't provided
        if X is None:
            X = self.get_processed_image_paths(balanced=balanced, max_num=dataset_size)
        if Y is None:
            Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths


        #splits into training, validation, and test datasets
        X_train, X_validate, X_test = self.data_handler.split_data(X, train_ratio, val_ratio)

        print("X_train size: "+str(len(X_train)))
        print("X_validate size: "+str(len(X_validate)))


        if model_arch.lower() == "cnn": 
            classifier = self.create_CNN()  
        elif model_arch.lower() == "unet":
            classifier = self.create_Unet()
        else:
            print("Unsupported specified model type")
            return


        training_generator = self.create_data_generator(X_train, Y, batch_size, "train")

        # patient early stopping
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

        # fits the model on batches with real-time data augmentation:
        classifier.fit_generator(generator=training_generator,
                                steps_per_epoch=len(X_train) / batch_size,
                                epochs=epochs,
                                validation_data=self.create_data_generator(X_validate, Y, batch_size, "test"), 
                                callbacks = [early_stopping])


        classifier.save("./trained_models/"+str(self.get_model_arch_filename_prefix(model_arch))+"_model.h5")

        return classifier, X_train, X_validate, X_test, Y


    #trains binary classifier with specified hyperparameters, then evaluates the results and saves to disk
    @abstractmethod
    def train_evaluate(self, classification_type="", model_arch="cnn", training_type="regular"):
        #loads hyperparameters because they might have been changed in the main menu
        self.hyperparameters = self.data_handler.load_hyperparameters()

        train_ratio = self.hyperparameters[classification_type][model_arch]['train_ratio']
        validation_ratio = self.hyperparameters[classification_type][model_arch]['val_ratio']
        batch_size = self.hyperparameters[classification_type][model_arch]['batch_size']
        epochs = self.hyperparameters[classification_type][model_arch]['epochs']
        dataset_size = self.hyperparameters[classification_type][model_arch]['dataset_size']
        balanced = self.hyperparameters[classification_type][model_arch]['balanced']
        n_splits = self.hyperparameters[classification_type][model_arch]['resampling_ensemble_n_splits']

        params = {"model_arch": model_arch, 
                    "dataset_size": dataset_size, 
                    "balanced": balanced,
                    "train_ratio": train_ratio,
                    "val_ratio": validation_ratio, 
                    "batch_size": batch_size, 
                    "epochs": epochs}


        if training_type == "regular":
            # #trains model, which returns the trained model and dataset segments
            # classifier, X_train, X_validate, X_test, Y = self.train(**params)
            classifiers, X_trains, X_validates, X_tests, Ys = self.resampling_ensemble(n_splits = 0, **params)
        elif training_type == "resampling_ensemble":
            classifiers, X_trains, X_validates, X_tests, Ys = self.resampling_ensemble(n_splits = n_splits, **params)



        # print("X_train: "+str(len(X_train[0])))
        # print("X_validate: "+str(len(X_validate[0])))
        # print("X_test: "+str(len(X_test[0])))
        # print("Y: "+str(len(Y[0])))



        #performs prediction on training portion
        print("--Training prediction analysis--")
        training_prediction_analysis = []
        try:
            for i in range(0, len(classifiers)):
                training_prediction_analysis.append(self.prediction_analysis(classifier=classifiers[i], feature_dataset=X_trains[i], full_label_dataset=Ys[i], batch_size=batch_size))
        except Exception as error:
            print("Couldn't perform training prediction analysis.")
            print(error)


        #perform prediction on validation portion
        print("--Validation prediction analysis--")
        validation_prediction_analysis = []
        try:
            for i in range(0, len(classifiers)):
                validation_prediction_analysis.append(self.prediction_analysis(classifier=classifiers[i], feature_dataset=X_validates[i], full_label_dataset=Ys[i], batch_size=batch_size))
        except Exception as error:
            print("Couldn't perform validation prediction analysis.")
            print(error)



        self.save_training_session("binary", training_type, classifiers, model_arch, training_prediction_analysis, validation_prediction_analysis)

    #saves training session, like the trained model, hyperparameters, and results, for future review
    def save_training_session(self, classification_type, training_type, classifiers, model_arch, training_analysis, validation_analysis):

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

        session_dir = self.data_handler.create_new_training_session_dir(project=self.project, classification_type=classification_type, model_arch=model_arch)


        #saves models
        try:
            for x in range(0, len(classifiers)):
                classifiers[x].save(session_dir+"/"+str(self.get_model_arch_filename_prefix(model_arch))+"_model_"+str(x+1)+"_of_"+str(len(classifiers))+".h5")
        except Exception as error:
            print("Error, couldn't save model: "+str(error))

        #saves hyperparameters
        new_hyperparameter_path = session_dir+"/hyperparameters.json"
        self.data_handler.copy_hyperparameters(new_hyperparameter_path)

        #Saves training prediction analysis
        for x in range(0, len(training_analysis)):
            to_save = {"all_stats": training_analysis[x][0], "agg_stats": training_analysis[x][1]}
            self.data_handler.save_to_json(session_dir+"/train_stats_"+str(x+1)+"_of_"+str(len(training_analysis))+".json", to_save)


        #Saves Validation prediction analysis
        for x in range(0, len(validation_analysis)):
            to_save = {'all_stats': validation_analysis[x][0], 'agg_stats': validation_analysis[x][1]}
            self.data_handler.save_to_json(session_dir+"/validation_stats_"+str(x+1)+"_of_"+str(len(training_analysis))+".json", to_save)


    #performs predictions and subsequent statistic calculations on unofficial test dataset
    @abstractmethod
    def test(self, model_arch="cnn", dataset_size=100):
        max_images = dataset_size
        X = self.get_processed_image_paths(dataset_type="test", balanced=False, max_num=max_images)
        Y = self.data_handler.read_train_labels() #don't limit, because will use this for finding masks to train_dicom_paths



        #loads the model
        try:
            classifier = load_model("./trained_models/"+str(self.get_model_arch_filename_prefix(model_arch))+"_model.h5", 
                                    custom_objects={'dice_coef_loss': self.dice_coef_loss, 'dice_coef': self.dice_coef})
        except Exception as error:
            print(error)
            print("Model doesn't exist for "+str(model_arch))
            return


        #performs statistical analysis and returns the result
        results = self.prediction_analysis(classifier=classifier, feature_dataset=X, full_label_dataset=Y, batch_size=5, verbose=True)





"""

Handles binary classification training, validation, and testing

"""
class BinaryClassifier(Classifier):
    def __init__(self, project):
        super().__init__(project)


    def print_something(self):
        print("Something")

    #creates CNN sculpted for binary classification
    def create_CNN(self):
        CNN_size = self.hyperparameters['binary']['cnn']['conv_layer_size']
        num_conv_layers = self.hyperparameters['binary']['cnn']['num_conv_layers']
        pool_size = (self.hyperparameters['binary']['cnn']['pool_size'], self.hyperparameters['binary']['cnn']['pool_size'])
        filter_size = (self.hyperparameters['binary']['cnn']['filter_size'], self.hyperparameters['binary']['cnn']['filter_size'])
        CNN_activation = self.hyperparameters['binary']['cnn']['conv_activation']
        dense_activation = self.hyperparameters['binary']['cnn']['dense_activation']
        output_activation = self.hyperparameters['binary']['cnn']['output_activation']
        loss = self.hyperparameters['binary']['cnn']['loss']
        optimizer = self.hyperparameters['binary']['cnn']['optimizer']
        last_layer_size = self.hyperparameters['binary']['cnn']['last_layer_size']
        dropout = self.hyperparameters['binary']['cnn']['dropout']

        if loss=="dice_coef_loss":
            loss = self.dice_coef_loss




        classifier = Sequential()


        # Step 2 - Pooling
        #pooling uses a 2x2 or something grid (most of the time is 2x2), goes over the feature maps, and the largest values as its going over become the values in the pooled map
        #slides with a stride of 2. At the end, the pool map should be (length/2)x(width/2)


        classifier.add(Conv2D(CNN_size, filter_size, input_shape = (self.image_width, self.image_height, 1), padding="same", activation = CNN_activation))
        classifier.add(MaxPooling2D(pool_size = pool_size))
        # classifier.add(BatchNormalization(axis=3))
        # classifier.add(Dropout(dropout))


        #adds hidden convolutional layers
        for x in range(1, num_conv_layers):
            classifier.add(Conv2D(CNN_size, filter_size, padding="same", activation = CNN_activation))
            classifier.add(MaxPooling2D(pool_size = pool_size))
            # classifier.add(BatchNormalization(axis=3))
            # classifier.add(Dropout(dropout))



        #flattents the layers
        classifier.add(Flatten())

        #128 is an arbitrary number that can be decreased to lower computation time, and increased for better accuracy
        classifier.add(Dense(units = last_layer_size, activation = dense_activation))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(units = 1, activation = output_activation))

        classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
        # classifier.compile(optimizer = 'adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        classifier.summary()

        return classifier


    #creates and returns U-net architecture model
    def create_Unet(self):
        start_size = self.hyperparameters['binary']['unet']['start_size']
        pool_size = (self.hyperparameters['binary']['unet']['pool_size'],self.hyperparameters['binary']['unet']['pool_size'])
        filter_size = (self.hyperparameters['binary']['unet']['filter_size'],self.hyperparameters['binary']['unet']['filter_size'])
        conv_activation = self.hyperparameters['binary']['unet']['conv_activation']
        dense_activation = self.hyperparameters['binary']['unet']['dense_activation']
        output_activation = self.hyperparameters['binary']['unet']['output_activation']
        loss = self.hyperparameters['binary']['unet']['loss']
        optimizer = self.hyperparameters['binary']['unet']['optimizer']
        last_layer_size = self.hyperparameters['binary']['unet']['last_layer_size']

        if loss=="dice_coef_loss":
            loss = self.dice_coef_loss

        if optimizer=="adam":
            optimizer = Adam(lr=1e-5)

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
        dense1 = Dense(units = last_layer_size, activation = dense_activation)(dense1)

        dense2 = Dense(units = 1, activation = output_activation)(dense1)
        dense2 = Dropout(0.25)(dense2)

        
        model = Model(inputs=inputs, outputs=dense2)

        # model.compile(optimizer=Adam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        model.summary()

        return model

    #creates custom data generator specific for labels of type binary
    def create_data_generator(self, feature_dataset, label_dataset, batch_size, step="train"):
        params = self.get_data_generator_params(step=step)

        generator = DataGenerator(feature_dataset, label_dataset, batch_size, "binary", **params)
        return generator

    #returns list of size 1 of confusion matrix
    def calculate_confusion_matrices(self, target_data, prediction_data):

        prediction_data = np.array(prediction_data)

        # print("Target data: "+str(target_data.shape))
        # print("Prediction data: "+str(prediction_data.shape))

        #calculates confusion matrix
        conf_matrix = confusion_matrix(target_data, prediction_data)
        # print("Confusion matrix: ")
        # print(conf_matrix)

        return [conf_matrix]


    def train_evaluate(self, model_arch="cnn", training_type="regular"):
        super().train_evaluate(classification_type="binary", model_arch=model_arch, training_type=training_type)


    def test(self, model_arch):

        print()
        print("You can find the model you want to test by specifying a date and training session number.")
        print()
        print("Training session dates available:")

        training_session_dates = self.data_handler.get_training_session_dates(project=self.project, classification_type="binary", model_arch=model_arch)

        for x in range(0, len(training_session_dates)):
            date = training_session_dates[x]
            #if today, print it special
            if date == self.data_handler.get_today():
                print("  "+str(x+1)+") "+str(training_session_dates[x])+" (today)")
            else:
                print("  "+str(x+1)+") "+str(training_session_dates[x]))
        print()
        date_index = int(input("Choice: "))

        while date_index<0 or date_index>len(training_session_dates):
            print("Incorrect choice, please choose a number in the list. ")
            date_index = int(input("Choice: "))

        print("Date chosen: "+str(training_session_dates[date_index-1]))

        print(" -- To be implemented later --")




"""

Handles segmentation classification training, validation, and testing

"""
class SegmentationClassifier(Classifier):
    def __init__(self, project):
        super().__init__(project)

    #creates CNN sculpted for segmentation classification
    def create_CNN(self):

        CNN_size = self.hyperparameters['segmentation']['cnn']['conv_layer_size']
        pool_size = (self.hyperparameters['segmentation']['cnn']['pool_size'],self.hyperparameters['segmentation']['cnn']['pool_size'])
        filter_size = (self.hyperparameters['segmentation']['cnn']['filter_size'],self.hyperparameters['segmentation']['cnn']['filter_size'])
        conv_activation = self.hyperparameters['segmentation']['cnn']['conv_activation']
        dense_activation = self.hyperparameters['segmentation']['cnn']['dense_activation']
        output_activation = self.hyperparameters['segmentation']['cnn']['output_activation']
        dropout = self.hyperparameters['segmentation']['cnn']['dropout']
        loss = self.hyperparameters['segmentation']['cnn']['loss']
        optimizer = self.hyperparameters['segmentation']['cnn']['optimizer']

        if loss=="dice_coef_loss":
            loss = self.dice_coef_loss

        if optimizer=="adam":
            optimizer = Adam(lr=1e-5)




        # Initialising the CNN
        classifier = Sequential()

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
        dropout1 = Dropout(dropout)(pool1)

        conv2 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout1)
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)
        dropout2 = Dropout(dropout)(pool2)

        conv3 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout2)
        pool3 = MaxPooling2D(pool_size=pool_size)(conv3)
        dropout3 = Dropout(dropout)(pool3)

        conv4 = Conv2D(CNN_size, filter_size, activation=conv_activation, padding='same')(dropout3)
        pool4 = MaxPooling2D(pool_size=pool_size)(conv4)
        dropout4 = Dropout(dropout)(pool4)

        conv10 = Conv2D(1, (1, 1), activation=output_activation)(dropout4)
        # conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)


        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=optimizer, loss=loss, metrics=[self.dice_coef])

        model.summary()

        return model


    #creates and returns U-net architecture model specific for segmentation prediction
    def create_Unet(self):
        start_size = self.hyperparameters['segmentation']['unet']['start_size']
        pool_size = (self.hyperparameters['segmentation']['unet']['pool_size'],self.hyperparameters['segmentation']['unet']['pool_size'])
        filter_size = (self.hyperparameters['segmentation']['unet']['filter_size'],self.hyperparameters['segmentation']['unet']['filter_size'])
        conv_activation = self.hyperparameters['segmentation']['unet']['conv_activation']
        output_activation = self.hyperparameters['segmentation']['unet']['output_activation']
        loss = self.hyperparameters['segmentation']['unet']['loss']
        optimizer = self.hyperparameters['segmentation']['unet']['optimizer']
        last_layer_size = self.hyperparameters['segmentation']['unet']['last_layer_size']


        if loss=="dice_coef_loss":
            loss = self.dice_coef_loss

        if optimizer=="adam":
            optimizer = Adam(lr=1e-5)


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

        conv10 = Conv2D(last_layer_size, (1, 1), activation=output_activation)(conv9)
        # conv10 = Convolution2D(1, (1, 1), activation=conv_activation)(conv9)


        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=optimizer, loss=loss, metrics=[self.dice_coef])

        model.summary()

        return model


    #creates custom data generator specific for labels of type segments
    def create_data_generator(self, feature_dataset, label_dataset, batch_size, step="train"):
        params = self.get_data_generator_params(step=step)

        #can't easily augment, because that'll change mask/label positions
        params['augment'] = False

        generator = DataGenerator(feature_dataset, label_dataset, batch_size, "segment", **params)
        return generator

    #returns list of size 1 of confusion matrix
    def calculate_confusion_matrices(self, target_data, prediction_data):


        prediction_data = np.array(prediction_data)

        #calculates confusion matrix on each image for the pixels
        confusion_matrices = []
        for x in range(0, target_data.shape[0]):

            #flattens 512x512x1 True/False 2D array into 262144 list of True/False
            targets = target_data[x].flatten()
            predictions = prediction_data[x].flatten()

            #calculates confusion matrix
            conf_matrix = confusion_matrix(targets, predictions)

            confusion_matrices.append(conf_matrix)

        return confusion_matrices

    #trains segmentation with specified hyperparameters
    def train_evaluate(self, model_arch="cnn", training_type="regular"):
        super().train_evaluate(classification_type="segmentation", model_arch=model_arch, training_type=training_type)



    def test(self, model_arch):

        print()
        print("You can find the model you want to test by specifying a date and training session number.")
        print()
        print("Training session dates available:")

        training_session_dates = self.data_handler.get_training_session_dates(project=self.project, classification_type="binary", model_arch=model_arch)

        for x in range(0, len(training_session_dates)):
            date = training_session_dates[x]
            #if today, print it special
            if date == self.data_handler.get_today():
                print("  "+str(x+1)+") "+str(training_session_dates[x])+" (today)")
            else:
                print("  "+str(x+1)+") "+str(training_session_dates[x]))
        print()
        date_index = int(input("Choice: "))

        while date_index<0 or date_index>len(training_session_dates):
            print("Incorrect choice, please choose a number in the list. ")
            date_index = int(input("Choice: "))

        print("Date chosen: "+str(training_session_dates[date_index-1]))

        print(" -- To be implemented later --")





















if __name__=="__main__":
    classifier = BinaryClassifier("chest_radiograph")
    # classifier = SegmentationClassifier()

    # CNN_classifier.train(dataset_size=200)

    classifier.train(model_arch="cnn", dataset_size=10)
    # classifier.test(model_arch="unet", dataset_size=10)