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