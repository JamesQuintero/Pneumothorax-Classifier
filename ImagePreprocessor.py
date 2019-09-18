from DICOM_reader import DICOMReader
from DataHandler import DataHandler

import sys
import os

#ML libraries
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import cv2 #py -3 -m pip install opencv-python

"""

Handles image preprocessing, validation, and testing

"""
class ImagePreprocessor:

    image_width = 1024
    image_height = 1024

    dicom_reader = None
    data_handler = None

    def __init__(self):
        self.dicom_reader = DICOMReader()
        self.data_handler = DataHandler()



    #normalized based off train data, then applies to validate and test data
    #returns 4-tuple of normalized train, validate, and test, and scaler object for saving
    def normalize_data(self, train, validate, test, scaler_type="standard_scaler"):

        #if normalizing to fit a distribution curve
        if scaler_type.lower() == "standard_scaler":
            scaler = StandardScaler()
        #if normalizing to be between 0 and 1
        elif scaler_type.lower() == "0_1_scaler":
            scaler = MinMaxScaler(feature_range=(0,1))


        # print()
        # print(train)
        # print()
        # print(train[0])
        # print()
        # print(train[0][0])
        # print()
        # print(train[0][0][0])
        # print()
        # print(train[0][0][0][0])

        # print(train.shape)


        #https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix

        # #fits scalar to training data
        # for x in range(0, len(train)):
        #     fit_concat_training = scaler.fit(train[x])





        train_normalized = train/255
        validate_normalized = validate/255
        test_normalized = test/255



        return train_normalized, validate_normalized, test_normalized, scaler

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
        # result = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)

        # for x in range(image1.shape[0]):
        #     for y in range(image1.shape[1]):
        #         #have to handle overflow
        #         if image1[x][y]>image2[x][y]:
        #             result[x][y] = image1[x][y] - image2[x][y]
        #         else:
        #             result[x][y] = 0

        # return result

        return cv2.subtract(image1, image2)


    #applies canny edge detector to the image as the preprocessing step
    #source: https://github.com/FienSoP/canny_edge_detector
    def canny_edge_detector(self, image, kernel_size=5, sigma=1, strong_pixel=255, weak_pixel=75, high_threshold=0.15, low_threshold=0.05):


        #blurs image
        image_smoothed = self.apply_gaussian_blur(image, kernel_size, sigma)

        #blackens majority of image and whitens edges
        gradient_matrix, theta_matrix = self.sobel_filters(image_smoothed)

        #reduces white edges
        non_max_image = self.non_max_suppression(gradient_matrix, theta_matrix)

        #only considers important edges
        threshold_image = self.threshold(non_max_image, strong_pixel, weak_pixel, high_threshold, low_threshold)

        # #edge tracking
        edge_tracking = self.hysteresis(threshold_image, strong_pixel, weak_pixel)

        return edge_tracking

        # return non_max_image

    #reduces noise in an image by blurring
    def reduce_noise(self, image):
        try:
            return cv2.medianBlur(image,5)
        except:
            print("Error reducing noise in image.")
            return image

    #perfdorms filtering on the 2D image
    def edge_filter(self, image):

        #blurs image for noise reduction
        blurred = self.reduce_noise(image)

        # strong_pixel = 255 #255 default
        # weak_pixel = 127 #127 default
        # ret,threshold1 = cv2.threshold(blurred,weak_pixel,strong_pixel,cv2.THRESH_BINARY)


        #gets rid of the distinct white portions
        threshold1 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,2)
        threshold1 = self.reduce_noise(threshold1)
        #gets rid of smaller white portions
        threshold2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
        threshold2 = self.reduce_noise(threshold2)
        #gets rid of thinner white portions
        threshold3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5,2)
        threshold3 = self.reduce_noise(threshold3)

        result = self.subtract_images(image, threshold1)
        result = self.subtract_images(result, threshold2)
        result = self.subtract_images(result, threshold3)

        return result



    #crops image so that only lungs/ribcage are prominantly displayed. 
    def crop(self):
        pass




if __name__=="__main__":
    image_preprocessor = ImagePreprocessor()
    
    # image_preprocessor.normalize_data()