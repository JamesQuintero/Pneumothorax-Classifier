import pandas as pd #for reading data files

from mask_functions import rle2mask


"""
Handles label data reading, like image ID's and masks
"""
class DataHandler:

    train_labels_path = "./data/train_labels.csv"
    test_labels_path = "./data/test_labels.csv"


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


    #returns dataset split into train, validation, and test portions
    def split_data(self, dataset, train_ratio=0.5, validation_ratio=0.2):

        #improper ratios provided
        if train_ratio+validation_ratio>1:
            print("Invalid training and validation ratios provided to where they're >100%")
            return [], [], []


        test_ratio = 1-train_ratio-validation_ratio

        print("Train ratio: "+str(train_ratio))
        print("Validation ratio: "+str(validation_ratio))
        print("Test ratio: "+str(test_ratio))

        return [],[],[]