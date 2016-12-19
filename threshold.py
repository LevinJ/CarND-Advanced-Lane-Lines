import sys
import os
from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibrate import Calibrarte
import matplotlib.image as mpimg


class Threshold(Calibrarte):
    def __init__(self):

        return
    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         plt.imshow(gray, cmap='gray')
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
        # Return the result
        return binary_output

    def stack_image_horizontal(self, imgs):
        for i in range(len(imgs)):
            img = imgs[i]
            if len(img.shape) == 2:
                scaled_img = np.uint8(255*img/np.max(img))
                imgs[i] = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
                plt.imshow(imgs[i][...,::-1])
        res_img = np.concatenate(imgs, axis=1)
        return res_img
    def abs_sobel_thresh_one_image(self, fname):
        image = cv2.imread(fname)
        grad_binary = self.abs_sobel_thresh(image, orient='x', thresh_min=5, thresh_max=100)
        grad_binary_2 = self.abs_sobel_thresh(image, orient='y', thresh_min=5, thresh_max=100)
        
        res_img = self.stack_image_horizontal([image,grad_binary, grad_binary_2])
        return  res_img
    def run(self):
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg',
          './test_images/test5.jpg','./test_images/test6.jpg']
        res_imgs = []
        for fname in fnames:
            img = self.abs_sobel_thresh_one_image(fname)
            res_imgs.append(img)
       
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= Threshold()
    obj.run()