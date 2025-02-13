"""
James Quintero
Created: 2019
"""

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
        # print("2) Train/Test model")
        # print("3) View Train/Test results")
        print("2) Train model")
        print("3) View Training results")
        print("4) Test a trained model")

        print("0) Quit")
        choice = int(input("Choice: "))


        #Preprocess images
        if choice==1:
            self.preprocess()
        #Train/Test models
        elif choice==2:
            # self.classifier()
            self.train_menu()

        #View Train/Test results
        elif choice==3:
            self.view_results()

        #Test trained models
        elif choice==4:
            self.test_menu()


        #Quits the program
        elif choice==0:
            return False

        print()
        print()
        print()

        return True


    """
    Train model menu
    """
    def train_menu(self):

        while True:
            print()
            print()
            print("-- Train Menu --")
            print()

            classifier, classifier_type = self.initialize_classifier()
            if classifier is None:
                return

            print()


            model_arch = self.get_model_architecture()
            if model_arch == "":
                return


            print()


            #ask if they want to modify hyperparameters
            self.user_modify_hyperparameters(classifier_type, model_arch)

            print()

            training_type = self.get_model_training_type()
            if training_type == "":
                return


            classifier.train_evaluate(model_arch, training_type)


            print()
            print()
            print()
            print()
            to_continue = input("Continue Training/Testing? (y/n): ")
            if to_continue.lower()=="n" or to_continue.lower()=="no":
                break


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


    """
    Preprocessor menu
    """
    def preprocess(self):
        # self.image_preprocessor = ImagePreprocessor()
        self.image_preprocessor = ChestRadiograph()

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




    """
    View Train results
    """
    def view_results(self):
        # print()
        # print("# To be implemented later")
        # print()

        project = "chest_radiograph"

        while True:
            print()
            print()
            print("-- View Results Menu --")
            print()

            # classifier=0
            # classifier_type, model_arch, date_to_retrieve = 0

            classifier, classifier_type = self.initialize_classifier()
            if classifier is None:
                return

            print()
            

            model_arch = self.get_model_architecture()
            if model_arch == "":
                return


            print()



            #gets current day
            self.print_available_dates(project, classifier_type, model_arch)
            date_to_retrieve = self.get_date(project, classifier_type, model_arch)


            while date_to_retrieve!="-1":
                print()

                self.print_available_training_sessions(project, classifier_type, model_arch, date_to_retrieve)
                training_session_num = self.get_training_session_num(project, classifier_type, model_arch, date_to_retrieve)
                while training_session_num!=-1:


                    classifier.view_training_session_results(project, model_arch, date_to_retrieve, training_session_num)

                    print()
                    print()
                    print()
                    print()

                    #asks user for a new training session number
                    self.print_available_training_sessions(project, classifier_type, model_arch, date_to_retrieve)
                    training_session_num = self.get_training_session_num(project, classifier_type, model_arch, date_to_retrieve)
                
                #asks user for a new date
                self.print_available_dates(project, classifier_type, model_arch)
                date_to_retrieve = self.get_date(project, classifier_type, model_arch)


    """
    Test a trained model menu
    """
    def test_menu(self):
        project = "chest_radiograph"

        while True:
            print()
            print()
            print("-- Test Model Menu --")
            print()

            # classifier=0
            # classifier_type, model_arch, date_to_retrieve = 0

            classifier, classifier_type = self.initialize_classifier()
            if classifier is None:
                return

            print()
            

            model_arch = self.get_model_architecture()
            if model_arch == "":
                return


            print()



            #gets current day
            self.print_available_dates(project, classifier_type, model_arch)
            date_to_retrieve = str(self.get_date(project, classifier_type, model_arch))


            while date_to_retrieve!="-1":
                print()

                self.print_available_training_sessions(project, classifier_type, model_arch, date_to_retrieve)
                training_session_num = self.get_training_session_num(project, classifier_type, model_arch, date_to_retrieve)
                while training_session_num!=-1:

                    dataset_size = int(input("Test dataset size: "))

                    classifier.test(project, model_arch, date_to_retrieve, training_session_num, dataset_size)

                    print()
                    print()
                    print()
                    print()

                    #asks user for a new training session number
                    self.print_available_training_sessions(project, classifier_type, model_arch, date_to_retrieve)
                    training_session_num = self.get_training_session_num(project, classifier_type, model_arch, date_to_retrieve)
                
                #asks user for a new date
                self.print_available_dates(project, classifier_type, model_arch)
                date_to_retrieve = str(self.get_date(project, classifier_type, model_arch))






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

    def initialize_classifier(self):
        print("Classification type: ")
        print("1) Binary (Predicting positive or negative)")
        print("2) Segmentation (Predicting segments)")
        print("0) Quit")

        classifier_choice = int(input("Choice: "))

        if classifier_choice == 1:
            classifier = BinaryClassifier("chest_radiograph")
            classifier_type = "binary"
        elif classifier_choice==2:
            classifier = SegmentationClassifier("chest_radiograph")
            classifier_type = "segmentation"
        #user wants to quit
        elif classifier_choice==0:
            return None, ""
        #user input invalid menu choice
        else:
            print("Improper classification type")
            return None, ""

        return classifier, classifier_type


    def get_model_building_step(self):
        print("Model building step: ")
        print("1) Train")
        print("2) Test")
        print("0) Quit")

        choice = int(input("Choice: "))

        step = ""
        if choice==1:
            step = "train"
        elif choice==2:
            step = "test"
        elif choice==0:
            step = ""
        else:
            print("Improper model building step")
            
        return step


    def get_model_architecture(self):
        print("Model architecture: ")
        print("1) CNN")
        print("2) U-net")
        print("0) Quit")

        model_arch_choice = int(input("Choice: "))

        model_arch = ""
        if model_arch_choice==1:
            model_arch = "cnn"
        elif model_arch_choice==2:
            model_arch = "unet"
        elif model_arch_choice==0:
            model_arch = ""
        else:
            print("Improper model architecture")
            

        return model_arch

    #returns datetime object corresponding to user provided date
    def get_date(self, project, classification_type, model_arch):

        prompt = "Date to view (YYYY-MM-DD, -1 to quit): "
        date_to_view = input(prompt)
        # date_to_view = "2019-10-22"


        while True and date_to_view!="-1":

            date_to_view = self.data_handler.string_to_date(date_to_view)
            if date_to_view == "":
                print("Please try again, but with proper date format")
                date_to_view = input(prompt)
            #if successful format, just stop
            else:
                break

        #gets list of avialable dates
        available_dates = self.data_handler.get_training_session_dates(project, classification_type, model_arch)
        if date_to_view!="-1" and str(date_to_view) not in available_dates:
            print("Please pick an available date.")
            return self.get_date(project, classification_type, model_arch)


        return date_to_view



    #retrieves possible dates of the training sessions run under project, training_type, and model architecture
    def print_available_dates(self, project, training_type, model_arch):
        print()
        print("- Available training session dates -")
        available_dates = self.data_handler.get_training_session_dates(project=project, classification_type=training_type, model_arch=model_arch)

        for date in available_dates:
            print(date)


    """
    Prints the available training session numbers under date_to_retrieve
    """
    def print_available_training_sessions(self, project, classification_type, model_arch, date_to_retrieve):
        print()
        print("- Available training session numbers -")
        available_training_sessions = self.data_handler.get_training_session_numbers(project, classification_type, model_arch, date_to_retrieve)

        for training_session in available_training_sessions:
            print(training_session)



    #get user input for training session number under project, training_type, model architecture, and date
    def get_training_session_num(self, project, classifier_type, model_arch, date_to_retrieve):

        # available_training_session_numbers = self.data_handler.

        try:
            training_session_number = int(input("Training session number (-1 to quit): "))
        except Exception as error:
            training_session_number = -1

        return training_session_number



    #allows the user to modify hyperparameters json file
    def user_modify_hyperparameters(self, classifier_type, model_arch):
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

            

    def get_model_training_type(self):
        print("Model training type: ")
        print("1) Standard")
        print("2) Resample Ensembling")
        print("3) K-fold cross validation")
        print("4) Model averaging")
        print("5) Bagging (Bootstrapping Aggregation)")
        print("0) Quit")
        choice = int(input("Choice: "))

        training_type = "regular"
        if choice==2:
            training_type = "resampling_ensemble"
        elif choice==3:
            training_type = "kfold_cross_validation"
        elif choice==4:
            training_type = "weighted_model_averaging"
        elif choice==5:
            training_type = "bagging"

        #if user quits
        elif choice==0:
            training_type = ""

        return training_type



        

if __name__=="__main__":
    main = Main()

    success = True
    while success:
        success = main.menu()
