import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibrate import Calibrarte
import matplotlib.image as mpimg
from frametracking import g_frame_tracking


class Threshold(Calibrarte):
    def __init__(self):

        return
    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    def stack_image_horizontal(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image_horizontal(imgs, axis = 1, max_img_width = max_img_width, max_img_height=max_img_height)
    def stack_image_vertical(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image_horizontal(imgs, axis = 0, max_img_width = max_img_width, max_img_height=max_img_height)
    def __stack_image_horizontal(self, imgs, axis = 1, max_img_width = None, max_img_height=None):
        #first let's make sure all the imge has same size
        img_sizes = np.empty([len(imgs), 2], dtype=int)
        for i in range(len(imgs)):
            img = imgs[i]
            img_sizes[i] = np.asarray(img.shape[:2])
        if max_img_width is None:
            max_img_width = img_sizes[:,1].max()
        if max_img_height is None:
            max_img_height = img_sizes[:,0].max()
        for i in range(len(imgs)):
            img = imgs[i]
            img_width = img.shape[1]
            img_height = img.shape[0]
            if (img_width == max_img_width) and (img_height == max_img_height):
                continue
            imgs[i] = cv2.resize(img, (max_img_width,max_img_height))
            
            
        
        for i in range(len(imgs)):
            img = imgs[i]
            if len(img.shape) == 2:
                scaled_img = np.uint8(255*img/np.max(img))
                imgs[i] = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
#                 plt.imshow(imgs[i][...,::-1])
        res_img = np.concatenate(imgs, axis=axis)
        return res_img
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
    
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Calculate the magnitude 
        # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        # 6) Create a binary mask where mag thresholds are met
        # 7) Return this mask as your binary_output image
    #     binary_output = np.copy(img) # Remove this line
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
        # Return the binary image
        return binary_output
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        # Here I'm suppressing annoying error messages
        with np.errstate(divide='ignore', invalid='ignore'):
            absgraddir = np.absolute(np.arctan(sobely/sobelx))
            binary_output =  np.zeros_like(absgraddir)
            binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        
        return binary_output
    def thresh_one_image_fname(self, fname, debug=False):
        original_image = cv2.imread(fname)
        return self.__thresh_one_image(original_image, debug=debug)
    
    def __thresh_one_image(self, original_image, debug=False):
        image = self.undistort(original_image)

        res_imgs = []
        res_imgs.append(image)
        
        
        gradx = self.abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
        res_imgs.append(gradx)
        
        grady = self.abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
        res_imgs.append(grady)
        
        mag_binary = self.mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
        res_imgs.append(mag_binary)
        
        dir_binary = self.dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
        res_imgs.append(dir_binary)
        
        
        red_color = self.red_thres(image, (200, 255))
        res_imgs.append(red_color)
        
#         s_color = self.hls_thres(image, (90, 255))
        s_color = self.hsv_thres_yellow(image)
        res_imgs.append(s_color)
        
       
        
        combined = np.zeros_like(s_color)
#         combined[((gradx == 1) | (s_color == 1)) & (dir_binary == 1)] = 1
#         combined[(gradx == 1) | (s_color == 1)] = 1
#         half_width = combined.shape[1]
        combined = np.concatenate([s_color[:,:640], gradx[:,640:]],axis=1)
        res_imgs.append(combined)
        
        # region of interest
        
        
        color_combined = np.dstack(( np.uint8(255*gradx/np.max(gradx)), np.uint8(255*s_color/np.max(s_color)), np.zeros_like(s_color)))
        
        if g_frame_tracking.use_last_lane_area_as_roi():
            roi_img,color_combined = self.__region_of_interest_last_lane_area(res_imgs, color_combined, combined)
        else:
            roi_img,color_combined = self.__region_of_interest_fixed(res_imgs, color_combined, combined)
#         roi_img,color_combined = self.__region_of_interest_fixed(res_imgs, color_combined, combined)
        
        res_img = self.stack_image_horizontal(res_imgs)
        
        if not debug:
            return original_image, image, color_combined, roi_img
        return  res_img
    
    def thresh_one_image(self, original_image, debug=False):
        #input is an RGB image
        return self.__thresh_one_image(original_image, debug=debug)
    def __region_of_interest_last_lane_area(self, res_imgs,color_combined, combined):
        mask = g_frame_tracking.last_roi
#         plt.imshow(mask)
        color_combined = cv2.bitwise_and(color_combined, mask)
#         plt.imshow(color_combined[...,::-1])
        res_imgs.append(color_combined)
        
        roi_img = cv2.bitwise_and(combined, mask[:,:,0])
        res_imgs.append(roi_img)
        return roi_img,color_combined
    def __region_of_interest_fixed(self, res_imgs,color_combined, combined):
        top_x_gap = 50
        bottom_x_gap = 150
        vertices = np.array([[(600-top_x_gap,451), (690 + top_x_gap,451), (1165 + bottom_x_gap,728),(240-bottom_x_gap,728)]], dtype=np.int32)
        cv2.polylines(color_combined, vertices, color=(0,0,255),isClosed=True,thickness=6)
        res_imgs.append(color_combined)
        
        roi_img = self.__region_of_interest(combined, vertices)
        res_imgs.append(roi_img)
        return roi_img,color_combined
    def __region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def hls_thres(self, image, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#         H = hls[:,:,0]
#         L = hls[:,:,1]
        S = hls[:,:,2]
        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1
        return binary
    def format_coord(self, x, y):
        pt = self.hsv_img[y, x, :]
        return 'HSV value, x={:.0f}, y={:.0f}  [h={}, s={}, v={}]'.format(x, y, pt[0],pt[1],pt[2])
    
    def hsv_thres_yellow(self, image):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lowerb = np.array([20, 40, 100], dtype=np.uint8)
        upperb = np.array([30, 255, 255], dtype=np.uint8)
        

        
        
        mask = cv2.inRange(hsv_img, lowerb, upperb)
        mask[mask==255]=1
        #for the purpose of showing hsv value
#         self.hsv_img = hsv_img 
#         _, ax = plt.subplots()
#         ax.format_coord = self.format_coord
#         ax.imshow(image[...,::-1])

        return mask
    
    def red_thres(self, image, thresh=(0, 255)):
        R = image[:,:,2]
        binary = np.zeros_like(R)
        binary[(R > thresh[0]) & (R <= thresh[1])] = 1
        return binary
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
          './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
#         fnames = ['./test_images/challenge2.jpg']
#         fnames = ['./exception_img.jpg']
        res_imgs = []
        for fname in fnames:
            img = self.thresh_one_image_fname(fname,debug=True)
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