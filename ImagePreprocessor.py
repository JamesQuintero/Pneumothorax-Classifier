"""
James Quintero
Created: 2019
"""


from DICOM_reader import DICOMReader
from DataHandler import DataHandler

import sys
import os

#ML libraries
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy import misc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import cv2 #py -3 -m pip install opencv-python
import matplotlib.pyplot as plt #for displaying DICOM files
from PIL import Image

#for abstraction
from abc import ABC, abstractmethod

"""

Abstract class for handling image preprocessing

"""
class ImagePreprocessor(ABC):

    image_width = 1024
    image_height = 1024

    preprocessed_ext = "png"

    dicom_reader = None
    data_handler = None

    def __init__(self):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()



    #normalized based off train data, then applies to validate and test data
    #returns 4-tuple of normalized train, validate, and test, and scaler object for saving
    def normalize_data(self, train, validate, test, scaler_type=None):

        #if normalizing to fit a distribution curve
        if scaler_type.lower() == "standard_scaler":
            scaler = StandardScaler()
        #if normalizing to be between 0 and 1
        elif scaler_type.lower() == "0_1_scaler":
            scaler = MinMaxScaler(feature_range=(0,1))


        # train_normalized = train/255
        # validate_normalized = validate/255
        # test_normalized = test/255



        mean = np.mean(train)
        std = np.std(train)

        # Subtract it equally from all splits
        train_normalized = (train - mean) / std
        validate_normalized = (validate - mean)/std
        test_normalized = (test - mean)/std


        return train_normalized, validate_normalized, test_normalized, scaler

    #normalizes single list of images
    def normalize_data(self, images):
        return images/255

    #applies gaussian blur to the provided image, and returns it
    def apply_gaussian_blur(self, image, kernel_size=5, sigma=1):
        #cuts down image
        was_expanded = False
        if len(image.shape)>=3:
            image = np.squeeze(image, axis=2)
            was_expanded = True


        image_blurred = convolve(image, self.gaussian_kernel(kernel_size, sigma))

        #re-expand if image was originally expanded
        if was_expanded:
            image_blurred = np.expand_dims(image_blurred, axis=2)

        return image_blurred


    #source: https://github.com/FienSoP/canny_edge_detector
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    #Gradient detection (blackening)
    def sobel_filters(self, image):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(image, Kx)
        Iy = ndimage.filters.convolve(image, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    #thins out edges
    def non_max_suppression(self, gradient_matrix, theta_matrix):
        M, N = gradient_matrix.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = theta_matrix * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = gradient_matrix[i, j+1]
                        r = gradient_matrix[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = gradient_matrix[i+1, j-1]
                        r = gradient_matrix[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = gradient_matrix[i+1, j]
                        r = gradient_matrix[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = gradient_matrix[i-1, j-1]
                        r = gradient_matrix[i+1, j+1]

                    if (gradient_matrix[i,j] >= q) and (gradient_matrix[i,j] >= r):
                        Z[i,j] = gradient_matrix[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    #only considers important edges
    def threshold(self, image, strong_pixel=255, weak_pixel=75, high_threshold=0.15, low_threshold=0.05):

        # high_threshold = 0.15
        # low_threshold = 0.05
        # weak_pixel = 75
        # strong_pixel = 255

        highThreshold = image.max() * high_threshold;
        lowThreshold = highThreshold * low_threshold;

        M, N = image.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(weak_pixel)
        strong = np.int32(strong_pixel)

        strong_i, strong_j = np.where(image >= highThreshold)
        zeros_i, zeros_j = np.where(image < lowThreshold)

        weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img, strong_pixel=255, weak_pixel=75):

        M, N = img.shape
        weak = weak_pixel
        strong = strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img


    #each image is a 2D array of 8-bit values, so account for overflow
    def subtract_images(self, image1, image2):
        return cv2.subtract(image1, image2)


    #applies canny edge detector to the image as the preprocessing step
    #source: https://github.com/FienSoP/canny_edge_detector
    @abstractmethod
    def canny_edge_detector(self, image, kernel_size=5, sigma=1, strong_pixel=255, weak_pixel=75, high_threshold=0.15, low_threshold=0.05):
        #blurs image
        image_smoothed = self.apply_gaussian_blur(image, kernel_size, sigma)

        #blackens majority of image and whitens edges
        gradient_matrix, theta_matrix = self.sobel_filters(image_smoothed)

        #reduces white edges
        non_max_image = self.non_max_suppression(gradient_matrix, theta_matrix)

        #only considers important edges
        threshold_image = self.threshold(non_max_image, strong_pixel, weak_pixel, high_threshold, low_threshold)

        #edge tracking
        edge_tracking = self.hysteresis(threshold_image, strong_pixel, weak_pixel)


        threshold_image = self.subtract_images(image, edge_tracking)

        return threshold_image

        # return non_max_image

    #reduces noise in an image by blurring
    def reduce_noise(self, image):
        try:
            return cv2.medianBlur(image,5)
        except:
            print("Error reducing noise in image.")
            return image

    #perfdorms filtering on the 2D image
    @abstractmethod
    def edge_filter(self, image):

        #blurs image for noise reduction
        blurred = self.reduce_noise(image)
        # blurred = image

        #gets rid of the distinct white portions
        threshold1 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 21,2)
        threshold1 = self.reduce_noise(threshold1)
        #gets rid of smaller white portions
        threshold2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 11,2)
        threshold2 = self.reduce_noise(threshold2)
        #gets rid of thinner white portions
        threshold3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 3,2)
        threshold3 = self.reduce_noise(threshold3)

        result = image
        result = self.subtract_images(result, threshold1)
        result = self.subtract_images(result, threshold2)
        result = self.subtract_images(result, threshold3)

        return result


    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum or minimum filter depending on type
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    def detect_local_extremas(self, array, threshold=5, filter_type="min"):
        # define an connected neighborhood
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        neighborhood = morphology.generate_binary_structure(len(array.shape),2)
        

        # apply the local minimum filter; all locations of minimum value 
        # in their neighborhood are set to 1

        if filter_type.lower()=="min" or filter_type.lower()=="minimum":
            # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
            local_extreme = (filters.minimum_filter(array,size=threshold)==array)
        elif filter_type.lower()=="max" or filter_type.lower()=="maximum":
            local_extreme = (filters.maximum_filter(array,size=threshold)==array)


        # local_extreme is a mask that contains the peaks we are 
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.
        # 
        # we create the mask of the background
        background = (array==0)
        # 
        # a little technicality: we must erode the background in order to 
        # successfully subtract it from local_extreme, otherwise a line will 
        # appear along the background border (artifact of the local minimum filter)
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
        # 
        # we obtain the final mask, containing only peaks, 
        # by removing the background from the local_extreme mask
        detected_minima = local_extreme - eroded_background
        return np.where(detected_minima)


    #crops to focus on just the important parts of the radiograph
    @abstractmethod
    def crop(self, pixels):
        return pixels


    #performs bulk preprocessing on training images
    def bulk_preprocessing(self, dataset_type="train", replace=True):

        if dataset_type.lower() == "train":
            dicom_paths = self.dicom_reader.load_dicom_train_paths()
        elif dataset_type.lower() == "test":
            dicom_paths = self.dicom_reader.load_dicom_test_paths()
        else:
            dicom_paths = []


        # for i, image_path in enumerate(train_dicom_paths):
        for i in range(0, len(dicom_paths)):
            image_path = dicom_paths[i]

            dicom_image = self.dicom_reader.get_dicom_obj(image_path)

            #extracts image_id from the file path
            image_id = image_path.split('\\')[-1].replace(".dcm", "")
            print("Image id: "+str(image_id))


            if dataset_type.lower() == "train":
                new_path = self.dicom_reader.get_dicom_filtered_train_path()
            elif dataset_type.lower() == "test":
                new_path = self.dicom_reader.get_dicom_filtered_test_path()

            new_path += "/"+str(image_id)+"."+str(self.preprocessed_ext)

            #if shouldn't replace file, and if file exists, then skip preprocessing
            if replace==False and os.path.isfile(new_path):
                continue

            # #skip non-pneumothorax
            # masks = self.data_handler.find_masks(image_id)
            # if len(masks)==0:
            #     continue



            pixels = dicom_image.pixel_array

            pixels = self.preprocess(pixels)

            self.dicom_reader.plot_pixel_array(pixels)





            # kernel_size = 5
            # sigma = 2
            # strong_pixel = 255
            # weak_pixel = 75
            # high_threshold = 0.15
            # low_threshold = 0.05

            # # dcm_image = self.image_preprocessor.apply_gaussian_blur(dcm_image)
            # new_dcm_image = self.image_preprocessor.canny_edge_detector(dcm_image, kernel_size, sigma, strong_pixel, weak_pixel, high_threshold, low_threshold)




            #if the pixel data is reduced (e.g. a 512 x 512 image is collapsed to 256 x 256) then ds.Rows and ds.Columns should be set appropriately. 
            #https://github.com/pydicom/pydicom/issues/738
            # dicom_image.PixelData = dicom_image.pixel_array.tobytes()







            im = Image.fromarray(pixels)
            im.save(new_path)

            print("Preprocessed image "+str(i)+"/"+str(len(dicom_paths)))


    #Preprocesses a single image
    @abstractmethod
    def preprocess(self, pixels):
        return pixels





"""

ImagePreprocessor class for Chest X-rays

"""
class ChestRadiograph(ImagePreprocessor):


    def __init__(self):
        super().__init__()
        pass

    def print_something(self):
        print("Something")

        result = self.normalize_data(255)
        print(result)

        result = self.crop(5)
        print(result)


    #applies canny edge detector to the image as the preprocessing step
    def canny_edge_detector(self, image, kernel_size=5, sigma=1, strong_pixel=255, weak_pixel=75, high_threshold=0.15, low_threshold=0.05):
        
        ret, threshold_image = cv2.threshold(image, weak_pixel, strong_pixel, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        threshold_image = self.subtract_images(image, threshold_image)

        return threshold_image

        # return non_max_image

    def edge_filter(self, image):
        #blurs image for noise reduction
        blurred = self.reduce_noise(image)
        # blurred = image

        strong_pixel = 255 #255 default
        weak_pixel = 99 #127 default
        ret,threshold0 = cv2.threshold(blurred,weak_pixel,strong_pixel,cv2.THRESH_BINARY)


        # #gets rid of the distinct white portions
        # threshold1 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 21,2)
        # threshold1 = self.reduce_noise(threshold1)
        # #gets rid of smaller white portions
        # threshold2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 11,2)
        # threshold2 = self.reduce_noise(threshold2)
        # #gets rid of thinner white portions
        # threshold3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 3,2)
        # threshold3 = self.reduce_noise(threshold3)

        result = image
        result = self.subtract_images(image, threshold0)
        # result = self.subtract_images(result, threshold1)
        # result = self.subtract_images(result, threshold2)
        # result = self.subtract_images(result, threshold3)

        return result


    #crops to focus on just the ribcage/lungs
    def crop(self, pixels):

        left_ribs, right_ribs = self.crop_column_indices(pixels)

        #crops columns for easier row prediction
        pixels = pixels[:, left_ribs:right_ribs]

        print("Left ribs: "+str(left_ribs))
        print("Right ribgs: "+str(right_ribs))

        # self.dicom_reader.plot_pixel_array(pixels)
        # print()

        top_ribs, bottom_ribs = self.crop_row_indices(pixels)

        #crops rows for top and bottom of lungs
        pixels = pixels[top_ribs:bottom_ribs, :]

        print("top ribs: "+str(top_ribs))
        print("Bottom ribs: "+str(bottom_ribs))

        # self.dicom_reader.plot_pixel_array(pixels)
        # print()

        return pixels


    #returns 2 index values denoting edge of rib cage on left and right sides
    def crop_column_indices(self, pixels):

        #gets average of column's pixel intensity
        column_intensities = []
        for x in range(0, pixels.shape[0]):
            avg = np.average(pixels[:,x])
            column_intensities.append(avg)


        #converts to numpy array for easier manipulation
        column_intensities = np.array(column_intensities)

        # plt.plot(column_intensities)
        # plt.show()



        threshold = int(pixels.shape[1]*0.2) #20% of image height is the threshold for local extremas

        #gets numpy list of indices of local extremas
        local_max = self.detect_local_extremas(column_intensities, threshold, "max")[0]
        print("Local max: "+str(local_max))

        #gets the middle because of ribs. 
        middle = pixels.shape[1]/2
        print("Middle: "+str(middle))

        #gets index of closest maxima to the middle, to denote the ribs. 
        spine_index = -1
        for x in range(0, len(local_max)-1):
            #if middle is between these two maxes
            if local_max[x]<= middle and local_max[x+1]>middle:
                #if left max is closer than right max, then consider that the ribs
                if abs(local_max[x]-middle) < abs(local_max[x+1]-middle):
                    spine_index = x
                else:
                    spine_index = x+1

                break

        if spine_index==-1:
            spine_index = len(local_max)-1

        print("Spine location: "+str(local_max[spine_index]))

        #if couldn't accurately get maximas, then don't crop
        if spine_index==-1 or spine_index==len(local_max)-1:
            return 0, pixels.shape[1]

        #if couldn't find ribs left of spine
        if spine_index==0:
            left_ribs = 0
        else:
            left_ribs = local_max[spine_index-1]

        #if couldn't find ribs right of spine
        if spine_index==len(local_max)-1:
            right_ribs = pixels.shape[1]
        else:
            right_ribs = local_max[spine_index+1]

        print("Left ribs: "+str(left_ribs))
        print("Right ribs: "+str(right_ribs))

        return left_ribs, right_ribs

    #returns 2 index values denoting edge of rib cage on left and right sides
    def crop_row_indices(self, pixels):

        #gets average of column's pixel intensity
        row_intensities = []
        for x in range(0, pixels.shape[0]):
            avg = np.average(pixels[x,:])
            row_intensities.append(avg)


        #converts to numpy array for easier manipulation
        row_intensities = np.array(row_intensities)

        # plt.plot(row_intensities)
        # plt.show()

        

        threshold = int(pixels.shape[0]*0.2) #20% of image height is the threshold for local extremas

        #gets numpy list of indices of local extremas
        local_max = self.detect_local_extremas(row_intensities, threshold, "max")[0]
        local_min = self.detect_local_extremas(row_intensities, threshold, "min")[0]
        # print("Local max: "+str(local_max))
        # print("Local min: "+str(local_min))

        #only keep local max's that aren't 0 intensity
        new_local_max = []
        for x in range(0, local_max.shape[0]):
            if row_intensities[local_max[x]]!=0:
                new_local_max.append(local_max[x])
        local_max = np.array(new_local_max)

        #only keep local min's that aren't 0 intensity
        new_local_min = []
        for x in range(0, local_min.shape[0]):
            if row_intensities[local_min[x]]!=255:
                new_local_min.append(local_min[x])
        local_min = np.array(new_local_min)

        # print("Local max: "+str(local_max))
        # print("Local min: "+str(local_min))

        #couldn't find enough intensity local maxima/minima
        if local_max.shape[0]==0 or local_min.shape[0]==0:
            return 0, pixels.shape[0]




        #gets the middle because of lungs. 
        middle = pixels.shape[0]/2
        # print("Middle: "+str(middle))

        #gets index of closest maxima to the middle, to denote the ribs. 
        lungs_index = -1
        for x in range(0, len(local_min)-1):
            #if middle is between these two mins
            if local_min[x]<= middle and local_min[x+1]>middle:
                #if left max is closer than right max, then consider that the ribs
                # if abs(local_min[x]-middle) < abs(local_min[x+1]-middle):
                lungs_index = x
                # else:
                #     lungs_index = x+1

                break

        if lungs_index==-1:
            lungs_index = len(local_min)-1

        # print("Middle of lungs: "+str(local_min[lungs_index]))


        #After finding the middle of the lungs, we can find the top and bottom of the ribs by getting the nearest maxima to this minima
        top_ribs = 0
        bottom_ribs = 0
        for x in range(0, len(local_max)-1):
            if local_max[x] <= local_min[lungs_index] and local_max[x+1]>local_min[lungs_index]:
                # print("Found max: "+str(local_max[x])+" | "+str(local_max[+1]))
                top_ribs = local_max[x]
                #makes sure bottom of the ribs extend beyond middle of the image
                adding = 1
                while x+adding<len(local_max) and local_max[x+adding]<middle:
                    adding+=1
                bottom_ribs = local_max[x+adding]

                break

        #sets bottom ribsif not found
        if bottom_ribs==0:
            bottom_ribs = local_max[-1]

        # print("Top ribs: "+str(top_ribs))
        # print("Bottom ribs: "+str(bottom_ribs))

        #leeway for top of ribs
        top_ribs = max(0, top_ribs-int(pixels.shape[0]*0.05))


        return top_ribs, bottom_ribs


    def preprocess(self, pixels):
        #crops
        pixels = self.crop(pixels)

        #normalizes cropped image so that the blackest pixel is once again 255, 
        #and whitest is once again 0, which the rest ajusting accordingly
        # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # pixels = self.renormalize(pixels)

        # self.dicom_reader.plot_pixel_array(pixels)

        # ## Perform preprocessing ##
        # pixels = np.invert(pixels)

        #resizes to 512x512
        pixels = cv2.resize(pixels, (512, 512))


        intensity_mean = np.mean(pixels)
        intensity_median = np.median(pixels)

        print("Mean: "+str(intensity_mean))
        print("Median: "+str(intensity_median))




        # self.dicom_reader.plot_pixel_array(pixels)
        # pixels = self.edge_filter(pixels)

        strong_pixel = 255
        edged_image = self.canny_edge_detector(image=pixels, weak_pixel=75, strong_pixel=strong_pixel)

        #while the mean pixel intensity is too low, then lower the strength of the edge detector to include more pixels
        while np.mean(edged_image)<30 and strong_pixel>0:
            strong_pixel = max(0, strong_pixel-50)
            edged_image = self.canny_edge_detector(image=pixels, weak_pixel=75, strong_pixel=strong_pixel)


        pixels = edged_image

        new_intensity_mean = np.mean(pixels)
        new_intensity_median = np.median(pixels)

        print("Edged Mean: "+str(new_intensity_mean))
        print("Edged Median: "+str(new_intensity_median))


        #brighten image if its mean intensity is too low
        if intensity_mean<150:
            alpha = 2.0 # Simple contrast control default = 1.0
            beta = 0    # Simple brightness control default = 0.0


            pixels = cv2.convertScaleAbs(pixels, alpha=alpha, beta=beta)
            # for y in range(pixels.shape[0]):
            #     for x in range(pixels.shape[1]):
            #             pixels[y,x] = np.clip(alpha*pixels[y,x] + beta, 0, 255)

        # pixels = np.invert(pixels)

        # cv2.imshow('Original Image', pixels)
        # cv2.imshow('New Image', new_image)

        return pixels







if __name__=="__main__":

    chest_xray = ChestRadiograph()

    chest_xray.bulk_preprocessing("train", True)