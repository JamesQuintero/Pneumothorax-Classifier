from DataHandler import DataHandler
from ImagePreprocessor import ImagePreprocessor

import sys
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import imageio

"""

Retrieves and provides data in batches for large dataset deep learning

"""
class DataGenerator(keras.utils.Sequence):

    data_handler = None
    image_preprocessor = None

    #passes in max number of images
    #batch size is the number of items per batch
    def __init__(self, image_paths, labels, batch_size, dim=(1024, 1024, 1), shuffle=False):
        self.dim = dim
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_handler = DataHandler()
        self.image_preprocessor = ImagePreprocessor()

        ## For debugging ##
        # self.get_processed_images(0, batch_size)
    
    #returns number of batches
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))
  
  
    #loads processed images, and returns them in batch
    #returns the index'th batch size amount of items
    def __getitem__(self, index):

        start = self.batch_size * index
        end = self.batch_size * (index+1)

        batch_x, batch_y = self.get_processed_images(start, end)

        return batch_x, batch_y


    #returns list of images and list of their corresponding labels
    def get_processed_images(self, start, end):

        image_height = self.dim[0]
        image_width = self.dim[1]
        image_channels = self.dim[2]


        #initialize feature and target images to zeroes
        X_train = np.zeros(((end-start), image_height, image_width, image_channels), dtype=np.uint8) #is type 8-bit for pixel intensity values
        Y_train = np.zeros((end-start), dtype=np.uint8) #is type bool because if pixel is 1, there is mask, 0 otherwise

        sys.stdout.flush() #what is this for?


        for i in range(start, min(end, len(self.image_paths))):
            image_path = self.image_paths[i]
            png_image = imageio.imread(image_path)

            #extracts image_id from the file path
            image_id = self.image_paths[i].split('\\')[-1].replace("."+str(self.image_preprocessor.preprocessed_ext), "")
            # print("Image id: "+str(image_id))

            # #apply dicom pixels
            X_train[i-start] = np.expand_dims(png_image, axis=2)

            masks = self.data_handler.find_masks(image_id=image_id, dataset=self.labels)


            try:
                #if no masks for image, then skip
                if len(masks)==0:
                    continue
                else:
                    # if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
                    last_mask = masks[-1]

                    ## To do:
                    ##   Account for all masks found 
                    ## 

                    #converts to boolean
                    # Y_train[i] = np.expand_dims(rle2mask(last_mask, image_height, image_width), axis=2)


                    Y_train[i-start] = 1


            except KeyError:
                # Y_train[n] = np.zeros((image_height, image_width, 1)) # Assume missing masks are empty masks.
                Y_train[i-start] = 0

        X_train = self.image_preprocessor.normalize_data(X_train)

        return X_train, Y_train