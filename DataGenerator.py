"""
James Quintero
Created: 2019
"""

from DataHandler import DataHandler
from ImagePreprocessor import *

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
    augment = False

    #passes in max number of images
    #batch size is the number of items per batch
    def __init__(self, image_paths, labels, batch_size, label_type="binary", dim=(1024, 1024, 1), augment=False, shuffle=False):
        self.dim = dim
        self.image_paths = image_paths
        self.labels = labels
        self.label_type = label_type
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        self.data_handler = DataHandler()
        self.image_preprocessor = ChestRadiograph()

        ## For debugging ##
        self.get_processed_images(0, batch_size)
    
    #returns number of batches
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))
  
  
    #loads processed images, and returns them in batch
    #returns the index'th batch size amount of items
    def __getitem__(self, index):

        start = self.batch_size * index
        end = min(self.batch_size * (index+1), len(self.image_paths))

        batch_x, batch_y = self.get_processed_images(start, end)

        return batch_x, batch_y


    #returns list of images and list of their corresponding labels
    def get_processed_images(self, start, end):

        image_height = self.dim[0]
        image_width = self.dim[1]
        image_channels = self.dim[2]


        #initialize feature and target images to zeroes
        X_train = np.zeros(((end-start), image_height, image_width, image_channels), dtype=np.uint8) #is type 8-bit for pixel intensity values

        if self.label_type.lower() == "binary":
            Y_train = np.zeros((end-start), dtype=np.uint8) #is type bool because if pixel is 1, there is mask, 0 otherwise
        else:
            Y_train = np.zeros(((end-start), image_height, image_width, image_channels), dtype=np.bool) #is type bool because if pixel is 1, there is mask, 0 otherwise

        sys.stdout.flush()


        for i in range(start, min(end, len(self.image_paths))):
            image_path = self.image_paths[i]
            png_image = imageio.imread(image_path)

            #extracts image_id from the file path
            image_id = self.image_paths[i].split('\\')[-1].replace("."+str(self.image_preprocessor.preprocessed_ext), "")
            # print("Image id: "+str(image_id))


            n = i-start

            # #apply dicom pixels
            X_train[n] = np.expand_dims(png_image, axis=2)

            masks = self.data_handler.find_masks(image_id=image_id, dataset=self.labels)


            try:
                #if no masks for image, then skip
                if len(masks)==0:
                    if self.label_type.lower() == "binary":
                        continue
                    else:  
                        Y_train[n] = np.zeros((image_height, image_width, image_channels))
                else:
                    last_mask = masks[-1]

                    if self.label_type.lower() == "binary":
                        Y_train[n] = 1
                    else:
                        #if one mask
                        if len(masks)==1:
                            Y_train[n] = np.expand_dims(self.data_handler.rle2mask(masks[0], image_height, image_width), axis=2)
                        else:
                            Y_train[n] = np.zeros((image_height, image_width, image_channels))
                            for mask in masks:
                                Y_train[n] =  Y_train[n] + np.expand_dims(self.data_handler.rle2mask(mask, image_height, image_width), axis=2)


            except KeyError as error:
                print("Error: "+str(error))
                if self.label_type.lower() == "binary":
                    Y_train[n] = 0
                else:
                    Y_train[n] = np.zeros((image_height, image_width, image_channels)) # Assume missing masks are empty masks.


        #gets augmented images
        X_train, Y_train = self.augment_images(X_train, Y_train)

        #normalize images
        X_train = self.image_preprocessor.normalize_data(X_train)

        # print("X_train: "+str(X_train.shape))
        # print("Y_train: "+str(Y_train.shape))
        # input()

        return X_train, Y_train


    #returns list containing original images and augmented (rotated, sheared, mirror, etc) images
    def augment_images(self, images, labels):
        if self.augment==False:
            return images, labels

        image_return = []
        image_return.extend(images)

        label_return = []
        label_return.extend(labels)

        # construct the image generator for data augmentation
        #don't include zoom, because that might remove the pneumothorax in the already cropped images
        datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.0,
            horizontal_flip=True, fill_mode="nearest")

        datagen.fit(images)


        batches = 0
        for x_batch, y_batch in datagen.flow(images, labels, batch_size=self.batch_size):

            image_return.extend(x_batch)
            label_return.extend(y_batch)

            batches += 1
            if batches >= len(images) / self.batch_size:
                #have to break manually
                break


        image_return = np.array(image_return)
        label_return = np.array(label_return)

        return image_return, label_return