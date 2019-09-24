#scripts to use
import os

from ImagePreprocessor import ImagePreprocessor
from DICOM_reader import DICOMReader
from CNNClassifier import CNNClassifier
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
    image_preprocessor = None
    CNN_classifier = None

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
        print("2) CNN classifier")

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
        self.CNN_classifier = CNNClassifier()

        self.CNN_classifier.train()


    """
    Preprocessor menu
    """
    def preprocess(self):
        self.image_preprocessor = ImagePreprocessor()

        choice = input("Wish to replace existing preprocessed images? (y/n): ")

        replace = False
        if choice.lower()=="y":
            replace = True

        self.image_preprocessor.bulk_preprocessing(replace)


        

if __name__=="__main__":
    main = Main()

    success = True
    while success:
        success = main.menu()
