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
        print("2) Classifier")

        print("0) Quit")
        choice = int(input("Choice: "))


        #Preprocess images
        if choice==1:
            self.preprocess()
        elif choice==2:
            self.cnn_classifier()


        #Quits the program
        elif choice==0:
            return False

        print()
        print()
        print()

        return True


    """
    CNN Classifier menu
    """
    def cnn_classifier(self):

        print("Classification type: ")
        print("1) Binary classification")
        print("2) Segmentation classification")

        choice = int(input("Choice: "))

        if choice == 1:
            classifier = BinaryClassifier()
        elif choice==2:
            classifier = SegmentationClassifier()
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

        choice = int(input("Choice: "))

        model_arch = ""
        if choice==1:
            model_arch = "cnn"
        elif choice==2:
            model_arch = "unet"
        else:
            print("Improper model architecture")
            return


        print()
        dataset_size = int(input("Dataset size: "))

            # classifier.train()

        if step == "train":
            classifier.train(model_arch, dataset_size)
        elif step == "test":
            classifier.test(model_arch, dataset_size)


    """
    Preprocessor menu
    """
    def preprocess(self):
        self.image_preprocessor = ImagePreprocessor()

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


        

if __name__=="__main__":
    main = Main()

    success = True
    while success:
        success = main.menu()
