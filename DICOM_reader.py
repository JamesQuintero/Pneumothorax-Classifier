"""
James Quintero
Created: 2019
"""

import glob #for loading DICOM files from disk
import pydicom #for reading DICOM files
from pydicom.encaps import encapsulate
import matplotlib.pyplot as plt #for displaying DICOM files
import pandas as pd #for reading data files
import os
import shutil

from DataHandler import DataHandler


images_path = './data/stage_2_images/*.dcm'



"""

Handles DICOM data reading, along with masks

"""
class DICOMReader:

    data_handler = None

    #path variables
    dicom_train_path = "./data/dicom-images-train/"
    dicom_filtered_train_path = "./data/dicom-images-train-filtered/"
    dicom_test_path = "./data/dicom-images-test-unofficial/"
    dicom_filtered_test_path = "./data/dicom-images-test-unofficial-filtered/"

    def __init__(self):
        self.data_handler = DataHandler()


    #prints metadata of dicom file
    def print_dcm_info(self, dicom_obj):
        # print("Filename.........:", file_path)
        print("Storage type.....:", dicom_obj.SOPClassUID)
        # print()

        pat_name = dicom_obj.PatientName
        display_name = pat_name.family_name + ", " + pat_name.given_name
        print("Patient's name......:", display_name)
        print("Patient id..........:", dicom_obj.PatientID)
        print("Patient's Age.......:", dicom_obj.PatientAge)
        print("Patient's Sex.......:", dicom_obj.PatientSex)
        print("Modality............:", dicom_obj.Modality)
        print("Body Part Examined..:", dicom_obj.BodyPartExamined)
        print("View Position.......:", dicom_obj.ViewPosition)
        
        if 'PixelData' in dicom_obj:
            rows = int(dicom_obj.Rows)
            cols = int(dicom_obj.Columns)
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(rows=rows, cols=cols, size=len(dicom_obj.PixelData)))
            if 'PixelSpacing' in dicom_obj:
                print("Pixel spacing....:", dicom_obj.PixelSpacing)

    #image should already by a 2D list of single value for grayscale
    def plot_pixel_array(self, image, figsize=(10,10)):
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=plt.cm.bone)
        plt.show()


    #plots many of the DICOM images at once
    def plot_many_images(self):
        start = 5   # Starting index of images
        num_img = 4 # Total number of images to show

        fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
        dataset = glob.glob(images_path)

        print("Len dataset: "+str(len(dataset)))

        for q, file_path in enumerate(dataset[start:start+num_img]):

            print(str(q)+": "+str(file_path))
            dataset = pydicom.dcmread(file_path)
            #show_dcm_info(dataset)
            
            ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)




    def display_masks(self):
        start = 0   # Starting index of images
        num_img = 5 # Total number of images to show

        images_path = "./data/sample_train/*.dcm"
    
        df = self.data_handler.read_train_data()
        print("Num training labels: "+str(len(df)))

        #creates charts/images
        fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

        images = glob.glob(images_path)

        for q, file_path in enumerate(images[start:start+num_img]):
            dataset = pydicom.dcmread(file_path)

            #displays the body scan
            ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

            #extracts image_id from the file path
            image_id = file_path.split('\\')[-1].replace(".dcm", "")
            print(str(q)+" Imageid: "+str(image_id))
            self.show_dcm_info(dataset)

            #finds masks matching image_id
            found_masks = df[df['ImageId'].str.match(image_id)]
            found_masks = found_masks.values.tolist()


            

            #has multiple masks
            if len(found_masks)>1:
                print("Multiple masks")
                #adds all masks to plot
                for x in range(0, len(found_masks)):
                    mask = self.data_handler.rle2mask(found_masks[x][1], 1024, 1024).T

                ax[q].set_title('See Marker')
                ax[q].imshow(mask, alpha=0.3, cmap="Reds")

            #has single mask
            elif len(found_masks)==1 and found_masks[0][1] != '-1':
                print("Single mask")

                mask = self.data_handler.rle2mask(found_masks[0][1], 1024, 1024).T
                ax[q].set_title('See Marker')
                ax[q].imshow(mask, alpha=0.3, cmap="Reds")

            #has no masks
            else:
                print("Nothing to see")
                ax[q].set_title('Nothing to see')


            print()


        plt.show()



    #plots dicom_pixels and mask_pixels
    #dicom_pixels is a pixel_array
    def plot_dicom(self, dicom_pixels, mask_pixels):
        
        plt.figure(figsize=(10,10))

        #displays the body scan
        plt.imshow(dicom_pixels, cmap=plt.cm.bone)

        plt.set_title('See Marker')

        #displays the masks if there are any
        plt.imshow(mask_pixels, alpha=0.3, cmap="Reds")

        #displays everything
        plt.show()


    #returns list of paths that point to dicom training images
    def load_dicom_train_paths(self):
        try:
            train_fns = glob.glob(self.dicom_train_path+"/*/*/*.dcm")
        except Exception as error:
            print("load_dicom_train_paths() error: "+str(error))
            return []

        return train_fns

    #rerturns list of paths that point to filtered training images
    def load_filtered_dicom_train_paths(self):
        try:
            train_fns = glob.glob(self.dicom_filtered_train_path+"/*png")
        except Exception as error:
            print("load_filtered_dicom_train_paths() error: "+str(error))
            return []

        return train_fns

    #returns list of paths that point to dicom training images
    def load_dicom_test_paths(self):
        try:
            test_fns = glob.glob(self.dicom_test_path+"/*.dcm")
        except Exception as error:
            print("load_dicom_test_paths() error: "+str(error))
            return []

        return test_fns

    #rerturns list of paths that point to filtered training images
    def load_filtered_dicom_test_paths(self):
        try:
            test_fns = glob.glob(self.dicom_filtered_test_path+"/*png")
        except Exception as error:
            print("load_filtered_dicom_test_paths() error: "+str(error))
            return []

        return test_fns


    #returns dicom object stored at path
    def get_dicom_obj(self, path):


        ### This is commented out because on Windows, os.path.isfile returns false no matter what if the path is > 260 characters. 
        ### This is unpatched as of Python 3.7
        # #if the file doesn't exist, return None
        # if os.path.isfile(path)==False:
        #     print("File doesn't exist")
        #     return None

        try:
            # dataset = pydicom.read_file(path)
            dataset = pydicom.dcmread(path)
            return dataset
        except Exception as error:
            print("get_dicom_obj() error: "+str(error))
            return None


    #moves all dcm files from one dataset to the other, without the extra nested folders that causes problems on Windows
    def move_all_dcm_files(self, dataset_type_from, dataset_type_to):

        if dataset_type_from.lower() == "train":
            from_paths = self.load_dicom_train_paths()
        elif dataset_type_from.lower() == "test":
            from_paths = self.load_dicom_test_paths()

        if dataset_type_to.lower() == "train":
            to_path = self.dicom_train_path
        elif dataset_type_to.lower() == "test":
            to_path = self.dicom_test_path


        print("To path: "+str(to_path))
        for x in range(0, len(from_paths)):
            print("From path: "+str(from_paths[x]))

            try:
                #uses copy2 instead of copy so that metadata is also copied
                shutil.copy2(from_paths[x], to_path)
            except Exception as error:
                pass


    def extract_image_id(self, path, ext="dcm"):
        #extracts image_id from the file path
        try:
            return path.split('\\')[-1].replace("."+str(ext), "")
        except Exception as error:
            print("Error extracting image id: "+str(error))
            return ""


    def get_dicom_filtered_train_path(self):
        return self.dicom_filtered_train_path

    def get_dicom_filtered_test_path(self):
        return self.dicom_filtered_test_path



if __name__=="__main__":

    dicom_reader = DICOMReader()

    # dicom_reader.display_masks()
    dicom_reader.move_all_dcm_files("train", "test")