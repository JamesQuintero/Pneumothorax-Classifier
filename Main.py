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

    #global methods global variable
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

        print("0) Quit")
        choice = int(input("Choice: "))


        #Preprocess images
        if choice==1:
            self.preprocess()


        #Quits the program
        elif choice==0:
            return False

        print()
        print()
        print()

        return True


    """
    Preprocessor menu
    """
    def preprocess(self):
        pass

if __name__=="__main__":
    main = Main()

    success = True
    while success:
        success = main.menu()
