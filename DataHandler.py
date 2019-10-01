
import os
import pandas as pd #for reading data files
import json
from datetime import date
import shutil

import numpy as np

from mask_functions import rle2mask


"""
Handles any data handling, label data reading (image ID's and masks), hyperaparameters, etc. 
"""
class DataHandler:

    train_labels_path = "./data/train_labels.csv"
    test_labels_path = "./data/test_labels.csv"

    #hyperparameters file, used for training of models
    hyperparameters_path = "./trained_models/hyperparameters.json"

    training_session_dir = "./training_sessions/"


    def __init__(self):
        pass


    #returns training data as Panda DataFrame
    def read_train_labels(self):
        df = pd.read_csv(self.train_labels_path)
        return df


    #returns testing data as Panda DataFrame
    def read_test_labels(self):
        df = pd.read_csv(self.test_labels_path)
        return df

    #returns directory path to save training session to
    def get_training_session_path(self, project="chest_radiograph", classification_type="binary", model_arch="cnn", date="2019-9-26", training_session_num=1):
        return self.training_session_dir+"/"+str(project.lower())+"/"+str(classification_type.lower())+"/"+str(model_arch.lower())+"/"+str(date.lower())+"/"+str(training_session_num)


    #creates new training session dir in the proper directory in ./training_sessions
    def create_new_training_session_dir(self, project="", classification_type="", model_arch=""):

        #gets current date in iso 8061 format (YYYY-MM-DD)
        cur_date = self.get_today()

        training_session_path = self.get_training_session_path(project=project, 
                                                    classification_type=classification_type,
                                                    model_arch=model_arch, 
                                                    date=cur_date, 
                                                    training_session_num=1)

        # print("Training session path: "+str(training_session_path))

        #gets parent directory
        training_session_parent = training_session_path[:-1]

        # print("Training session parent: "+str(training_session_parent))
        
        #lists all training sessions for current day
        try:
            training_sessions = os.listdir(training_session_parent)

            #converts directories to ints
            training_sessions = [int(i) for i in training_sessions] 
            #sorts list
            training_sessions.sort()


            new_session_number = int(training_sessions[-1])+1
        except Exception as error:
            #training sessions don't exist for path, so start it
            # pass
            new_session_number = 1



        session_dir = self.create_new_training_session(project=project,
                                                            classification_type=classification_type,
                                                            model_arch=model_arch,
                                                            date=cur_date,
                                                            training_session_num=new_session_number)

        print("Session dir: "+str(session_dir))

        return session_dir

    #creates training session dir, and handles parent directories not existing
    def create_new_training_session(self, project="chest_radiograph", classification_type="binary", model_arch="cnn", date="2019-9-26", training_session_num=1):

        path = self.training_session_dir+"/"

        dirs = [project.lower(), classification_type.lower(), model_arch.lower(), date.lower(), training_session_num]

        #creates parent directories if need be
        for x in range(0, len(dirs)):

            if os.path.exists(path)==False:
                os.mkdir(path)

            path += str(dirs[x])+"/"


        #creates the training session
        os.mkdir(path)


        return path
        
    #returns list of dates that have training sessions under specified parameters
    def get_training_session_dates(self, project="chest_radiograph", classification_type="binary", model_arch="cnn"):

        path = self.training_session_dir+"/"+str(project.lower())+"/"+str(classification_type.lower())+"/"+str(model_arch.lower())

        dates = os.listdir(path)

        return dates


    #returns iso 8601 date format string of today
    def get_today(self):
        return str(date.today())


    def get_hyperparameters_path(self):
        return self.hyperparameters_path

    #returns hyperparameter dictionary from hyperparameter file
    def load_hyperparameters(self):

        if os.path.isfile(self.hyperparameters_path)==False:
            print("Error, hyperparameters file doesn't exist: "+str(self.hyperparameters_path))
            return {}

        try:
            with open(self.hyperparameters_path) as json_file:
                data = json.load(json_file)
                

                return data
        except Exception as error:
            print("Error, couldn't load hyperparameters: "+str(error))
            return {}

    #prints hyperparameter dictionary in a neat format
    def print_hyperparameters(self, hyperparameters):
        print(json.dumps(hyperparameters, indent=4, sort_keys=True))


    #saves hyperparameters dictionary to the json file
    def save_hyperparameters(self, hyperparameters):

        # hyper_parameters = {}
        # hyper_parameters['binary'] = {}
        # hyper_parameters['segmentation'] = {}

        # hyper_parameters['binary']['cnn'] = {}
        # hyper_parameters['binary']['unet'] = {}

        # hyper_parameters['segmentation']['cnn'] = {}
        # hyper_parameters['segmentation']['unet'] = {}

        # hyper_parameters['binary']['cnn']['train_ratio'] = 0.7
        # hyper_parameters['binary']['cnn']['val_ratio'] = 0.2
        # hyper_parameters['binary']['cnn']['dataset_size'] = 200
        # hyper_parameters['binary']['cnn']['batch_size'] = 10
        # hyper_parameters['binary']['cnn']['epochs'] = 10
        # hyper_parameters['binary']['cnn']['augmented'] = True
        # hyper_parameters['binary']['cnn']['conv_layer_size'] = 32
        # hyper_parameters['binary']['cnn']['pool_size'] = 3
        # hyper_parameters['binary']['cnn']['filter_size'] = 3
        # hyper_parameters['binary']['cnn']['conv_activation'] = "selu"
        # hyper_parameters['binary']['cnn']['dense_activation'] = "selu"
        # hyper_parameters['binary']['cnn']['output_activation'] = "sigmoid"
        # hyper_parameters['binary']['cnn']['loss'] = "mean_squared_error"
        # hyper_parameters['binary']['cnn']['optimizer'] = "adam"
        # hyper_parameters['binary']['cnn']['last_layer_size'] = 128
        # hyper_parameters['binary']['cnn']['dropout'] = 0.25

        # hyper_parameters['binary']['unet']['train_ratio'] = 0.7
        # hyper_parameters['binary']['unet']['val_ratio'] = 0.2
        # hyper_parameters['binary']['unet']['dataset_size'] = 100
        # hyper_parameters['binary']['unet']['batch_size'] = 10
        # hyper_parameters['binary']['unet']['epochs'] = 5
        # hyper_parameters['binary']['unet']['augmented'] = True
        # hyper_parameters['binary']['unet']['start_size'] = 16
        # hyper_parameters['binary']['unet']['depth'] = 3
        # hyper_parameters['binary']['unet']['pool_size'] = 2
        # hyper_parameters['binary']['unet']['filter_size'] = 3
        # hyper_parameters['binary']['unet']['conv_activation'] = "selu"
        # hyper_parameters['binary']['unet']['dense_activation'] = "selu"
        # hyper_parameters['binary']['unet']['output_activation'] = "sigmoid"
        # hyper_parameters['binary']['unet']['loss'] = "mean_squared_error"
        # hyper_parameters['binary']['unet']['optimizer'] = "adam"
        # hyper_parameters['binary']['unet']['last_layer_size'] = 128
        # hyper_parameters['binary']['unet']['dropout'] = 0



        # hyper_parameters['segmentation']['cnn']['train_ratio'] = 0.7
        # hyper_parameters['segmentation']['cnn']['val_ratio'] = 0.2
        # hyper_parameters['segmentation']['cnn']['dataset_size'] = 200
        # hyper_parameters['segmentation']['cnn']['batch_size'] = 10
        # hyper_parameters['segmentation']['cnn']['epochs'] = 10
        # hyper_parameters['segmentation']['cnn']['augmented'] = False
        # hyper_parameters['segmentation']['cnn']['conv_layer_size'] = 32
        # hyper_parameters['segmentation']['cnn']['pool_size'] = 3
        # hyper_parameters['segmentation']['cnn']['filter_size'] = 3
        # hyper_parameters['segmentation']['cnn']['conv_activation'] = "selu"
        # hyper_parameters['segmentation']['cnn']['dense_activation'] = "selu"
        # hyper_parameters['segmentation']['cnn']['output_activation'] = "sigmoid"
        # hyper_parameters['segmentation']['cnn']['loss'] = "mean_squared_error"
        # hyper_parameters['segmentation']['cnn']['optimizer'] = "adam"
        # hyper_parameters['segmentation']['cnn']['last_layer_size'] = 128
        # hyper_parameters['segmentation']['cnn']['dropout'] = 0.25

        # hyper_parameters['segmentation']['unet']['train_ratio'] = 0.7
        # hyper_parameters['segmentation']['unet']['val_ratio'] = 0.2
        # hyper_parameters['segmentation']['unet']['dataset_size'] = 30
        # hyper_parameters['segmentation']['unet']['batch_size'] = 10
        # hyper_parameters['segmentation']['unet']['epochs'] = 5
        # hyper_parameters['segmentation']['unet']['augmented'] = False
        # hyper_parameters['segmentation']['unet']['start_size'] = 16
        # hyper_parameters['segmentation']['unet']['depth'] = 3
        # hyper_parameters['segmentation']['unet']['pool_size'] = 2
        # hyper_parameters['segmentation']['unet']['filter_size'] = 3
        # hyper_parameters['segmentation']['unet']['conv_activation'] = "selu"
        # # hyper_parameters['segmentation']['unet']['dense_activation'] = ""
        # hyper_parameters['segmentation']['unet']['output_activation'] = "sigmoid"
        # hyper_parameters['segmentation']['unet']['loss'] = "dice_coef_loss"
        # hyper_parameters['segmentation']['unet']['optimizer'] = "adam"
        # hyper_parameters['segmentation']['unet']['last_layer_size'] = 1
        # hyper_parameters['segmentation']['unet']['dropout'] = 0

        self.save_to_json(self.hyperparameters_path, hyperparameters)



    #dumps data into json file to be located at path
    def save_to_json(self, path, data):
        #converts numpy numbers to python numbers
        def convert(number):
            if isinstance(number, np.int64):
                return int(number)
            elif isinstance(number, np.float64):
                return float(number)

        try:
            with open(path, 'w') as outfile:
                json.dump(data, outfile, default=convert, sort_keys=True, indent=4)
        except Exception as error:
            print("Error, couldn't save data to json: "+str(error))

    #copies hyperparameter file to new_path
    def copy_hyperparameters(self, new_path):
        shutil.copy2(self.hyperparameters_path, new_path)



    #returns list of mask coordinates in RLE format corresponding to image_id
    def find_masks(self, image_id, dataset=None):

        #if dataset isn't provided, default to whole training dataset
        if dataset is None:
            dataset = self.read_train_labels()

        masks = []

        #if dataset a dataframe
        if isinstance(dataset, pd.DataFrame):
            #finds masks matching image_id
            found_masks = dataset[dataset['ImageId'].str.match(image_id)]
            found_masks = found_masks.values.tolist()

            for row in found_masks:
                if row[1]!="-1":
                    masks.append(row[1])

        #if dataset is a list 
        elif isinstance(dataset, list):
            pass

        return masks



    #returns dataset split into train, validation, and test portions
    def split_data(self, dataset, train_ratio=0.5, validation_ratio=0.2):

        #improper ratios provided
        if train_ratio+validation_ratio>1:
            print("Invalid training and validation ratios provided to where they're >100%")
            return [], [], []


        test_ratio = 1-train_ratio-validation_ratio

        # print("Len: "+str(len(dataset)))
        # print("Train ratio: "+str(train_ratio))
        # print("Validation ratio: "+str(validation_ratio))
        # print("Test ratio: "+str(test_ratio))

        train_cutoff = int(len(dataset)*train_ratio) 
        val_cutoff = int(len(dataset)*validation_ratio) + train_cutoff 

        # print("train cutoff: "+str(train_cutoff))
        # print("Validation cutoff: "+str(val_cutoff))

        if train_cutoff<=0:
            return [], [], []


        training_set = dataset[ : train_cutoff]
        validation_set = dataset[train_cutoff : val_cutoff]
        testing_set = dataset[val_cutoff : ]

        return training_set, validation_set, testing_set

    #checks if string is a float
    def is_float(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    #checks if string is a float
    def is_int(self, string):
        try:
            int(string)
            return True
        except ValueError:
            return False


if __name__=="__main__":
    data_handler = DataHandler()

    # data_handler.save_hyperparameters({})
    hyperparameters = data_handler.load_hyperparameters()

    print(hyperparameters)