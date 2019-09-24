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
    augment = False

    #passes in max number of images
    #batch size is the number of items per batch
    def __init__(self, image_paths, labels, batch_size, dim=(1024, 1024, 1), augment=False, shuffle=False):
        self.dim = dim
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
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


        #gets augmented images
        X_train, Y_train = self.augment_images(X_train, Y_train)

        #normalize images
        X_train = self.image_preprocessor.normalize_data(X_train)

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
        datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")

        datagen.fit(images)


        batches = 0
        for x_batch, y_batch in datagen.flow(images, labels, batch_size=self.batch_size):
            # model.fit(x_batch, y_batch)

            image_return.extend(x_batch)
            label_return.extend(y_batch)

            # print("Batch "+str(batches))

            batches += 1
            if batches >= len(images) / self.batch_size:
                #have to break manually
                break


        image_return = np.array(image_return)
        label_return = np.array(label_return)

        # print("Original images: "+str(len(images)))
        # print("Original labels: "+str(len(labels)))
        # print(image_return.shape)
        # print("Total images: "+str(len(image_return)))
        # print("Total labels: "+str(len(label_return)))

        return image_return, label_return