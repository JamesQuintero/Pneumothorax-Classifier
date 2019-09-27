#scripts to use
import os

from ImagePreprocessor import ImagePreprocessor
from DICOM_reader import DICOMReader
from Classifier import *
from DataHandler import DataHandler

from multiprocessing import Process
import multiprocessing
# from queue import *


#GPU might be outdated, so just use CPU on Windows
# os.environ['CUDA_VISIBLE_DEVICES']='-1'


"""

Handles user input and utilizing all scripts in this project

"""
class Main: 

    data_handler = None

    def __init__(self):
        self.data_handler = DataHandler()



    """ 
    Interfaces this stock prediction program with the user
    """
    def menu(self):
        print("---- Pneumothorax Classifier ----")
        print()
        print("-- Menu --")
        print("1) Preprocess scans")
        print("2) Train/Test model")
        print("3) View Train/Test results")

        print("0) Quit")
        choice = int(input("Choice: "))


        #Preprocess images
        if choice==1:
            self.preprocess()
        #Train/Test models
        elif choice==2:
            self.classifier()
        #View Train/Test results
        elif choice==3:
            self.view_results()


        #Quits the program
        elif choice==0:
            return False

        print()
        print()
        print()

        return True


    """
    Classifier menu
    """
    def classifier(self):

        print()
        print()
        print("-- Classifier Menu --")
        print()

        print("Classification type: ")
        print("1) Binary classification")
        print("2) Segmentation classification")

        classifier_choice = int(input("Choice: "))

        if classifier_choice == 1:
            classifier = BinaryClassifier("chest_radiograph")
            classifier_type = "binary"
        elif classifier_choice==2:
            classifier = SegmentationClassifier("chest_radiograph")
            classifier_type = "segmentation"
        else:
            print("Improper classification type")
            return


        print()
        print("Model building step: ")
        print("1) Train")
        print("2) Test")

        choice = int(input("Choice: "))

        step = ""

        #user wants to train a model
        if choice==1:
            step = "train"
        elif choice==2:
            step = "test"
        else:
            print("Improper model building step")
            return


        print()
        print("Model architecture: ")
        print("1) CNN")
        print("2) U-net")

        model_arch_choice = int(input("Choice: "))

        model_arch = ""
        if model_arch_choice==1:
            model_arch = "cnn"
        elif model_arch_choice==2:
            model_arch = "unet"
        else:
            print("Improper model architecture")
            return


        # print()
        # dataset_size = int(input("Dataset size: "))



        if step == "train":
            #if user is trianing, ask if they want to modify hyperparameters
            print()
            print("Hyperparameters: ")
            self.print_hyperparameters(classifier_type, model_arch)
            

            print()
            choice = input("Modify? (y/n): ")

            while choice.lower()=="y" or choice.lower()=="yes":
                self.modify_hyperparameters(classifier_type, model_arch)

                print()
                print("New hyperparameters: ")
                self.print_hyperparameters(classifier_type, model_arch)

                print()
                choice = input("Continue modification? (y/n): ")


            classifier.train(model_arch)
        elif step == "test":
            classifier.test(model_arch)


    #allows user to modify hyperparameters 
    def modify_hyperparameters(self, classifier_type, model_arch):
        hyperparameters = self.data_handler.load_hyperparameters()

        #failed to modify hyperparameters due to incorrect classifier types or model architectures
        if classifier_type not in hyperparameters or model_arch not in hyperparameters[classifier_type]:
            return False


        key = input("Field to modify: ").lower()

        while key not in hyperparameters[classifier_type][model_arch]:
            print("Invalid key, please enter again. ") 
            key = input("Field to modify: ")

        new_value = input("New value: ")


        #Makes sure the user's new value has the same data type as the old value
        success = False
        if self.data_handler.is_int(new_value):
            if type(hyperparameters[classifier_type][model_arch][key])==int:
                new_value = int(new_value)
                success = True
        elif self.data_handler.is_float(new_value):
            if type(hyperparameters[classifier_type][model_arch][key])==float:
                new_value = float(new_value)
                success = True
        elif new_value.lower()=="false":
            if type(hyperparameters[classifier_type][model_arch][key])==bool:
                new_value = False
                success = True
        elif new_value.lower()=="true":
            if type(hyperparameters[classifier_type][model_arch][key])==bool:
                new_value = True
                success = True
        #previous value was a string
        else:
            if type(hyperparameters[classifier_type][model_arch][key])==str:
                new_value = str(new_value)
                success = True


        #couldn't successfully change value
        if success==False:
            print("Error, new value must be same datatype as old value.")
            return



        # #makes sure the new value matches the data type as the old value
        # if self.data_handler.is_int(new_value) and type(hyperparameters[classifier_type][model_arch][key])==int:
        #     new_value = int(new_value)
        # elif self.data_handler.is_float(new_value) and type(hyperparameters[classifier_type][model_arch][key])==float:
        #     new_value = float(new_value)
        # elif new_value.lower()=="false" and type(hyperparameters[classifier_type][model_arch][key])==bool:
        #     new_value = False
        # elif new_value.lower()=="true":
        #     new_value = True
        # else:
        #     print("Error, new value must be same datatype as old value.")
        #     return


        hyperparameters[classifier_type][model_arch][key] = new_value

        self.data_handler.save_hyperparameters(hyperparameters)


    def print_hyperparameters(self, classifier_type=None, model_arch=None):
        hyperparameters = self.data_handler.load_hyperparameters()


        #if classifier type isn't specified, print all hyperparameters
        if classifier_type==None:
            self.data_handler.print_hyperparameters(hyperparameters)
            return
        
        #if model arch isn't specified, print all model architectures for specified classifier type
        if model_arch==None:
            self.data_handler.print_hyperparameters(hyperparameters[classifier_type])

        #both classifier type and model architecture are specified, so print hyperparameters for it
        self.data_handler.print_hyperparameters(hyperparameters[classifier_type][model_arch])


    """
    Preprocessor menu
    """
    def preprocess(self):
        self.image_preprocessor = ImagePreprocessor()

        print()
        print()
        print("-- Preprocessor Menu --")
        print()


        print("Which dataset portion to preprocess?")
        print("1) Training dataset")
        print("2) Testing dataset")

        choice = int(input("Choice: "))
        if choice==1:
            dataset_type = "train"
        elif choice==2:
            dataset_type = "test"
        else:
            dataset_type = ""


        choice = input("Wish to replace existing preprocessed images? (y/n): ")

        replace = False
        if choice.lower()=="y":
            replace = True

        self.image_preprocessor.bulk_preprocessing(dataset_type=dataset_type, replace=replace)


    #View Train/Test results
    def view_results(self):
        pass


        

if __name__=="__main__":
    main = Main()

    success = True
    while success:
        success = main.menu()
