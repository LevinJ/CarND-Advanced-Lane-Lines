import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from threshold import Threshold


class BirdViewTransform(Threshold):
    def __init__(self):
        Threshold.__init__(self)
        return
    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients
    def corners_unwarp(self, undist):
        nx, ny = 9,6
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
#         plt.imshow(gray, cmap='gray')
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        if ret == True:
            # If we found corners, draw them! (just for fun)
#             cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # Choose an offset from image corners to plot detected corners
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])
    
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)
    
        # Return the resulting image and matrix
        return warped
    def bird_view(self, img):
        width, height = (img.shape[1], img.shape[0])
        
     
        # For source points I'm grabbing the outer four detected corners
#         src = np.float32([(603,451), (687,451), (1066,664),(330,664)])
        src = np.float32([(600,451), (680,451), (1165,728),(240,728)])
        
        
        
        
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        offset_x = 300
        offset_y = 0
        dst = np.float32([(offset_x, offset_y), (width-offset_x,offset_y), (width-offset_x,height), (offset_x, height)])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        lane_pixel_num = width- 2*offset_x
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, (width, height))
        
        #for debugging prupose
#         thickness = 3
#         pts = src.reshape((-1,1,2)).astype(np.int32)
#         cv2.polylines(img,[pts],True,1,thickness)
        
#         pts = dst.reshape((-1,1,2)).astype(np.int32)
#         cv2.polylines(warped,[pts],True,(0,0,255),thickness)
        
        return warped, Minv,lane_pixel_num
    def show_one_image(self,ax, fname):
        ax1,ax2,ax3 = ax
        img = cv2.imread(fname)
        undist = self.undistort(img)
        warped= self.bird_view(undist)
        
        img = img[...,::-1] #convert from opencv bgr to standard rgb
        undist = undist[...,::-1] #convert from opencv bgr to standard rgb
        warped = warped[...,::-1] #convert from opencv bgr to standard rgb
        ax1.imshow(img)
        ax1.set_title('Original Image')
          
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image')
        ax3.imshow(warped)
        ax3.set_title('Transformed Image')
        
        
        return
    def test_transform(self, fnames):
       

        
        _, axes = plt.subplots(len(fnames), 3,sharex=True)
        axes = axes.reshape(-1,3)
        for i in range(len(fnames)):
            fname = fnames[i]
            self.show_one_image(axes[i], fname)
      
        return
        
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
          './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
#         fnames = ['./exception_img.jpg']
#         self.test_transform(fnames)
        res_imgs = []
        for fname in fnames:
            original_img, img, color_combined, thres_img = self.thresh_one_image_fname(fname)
            pers_img, _ , _= self.bird_view(thres_img)
            res_imgs.append(self.stack_image_horizontal([original_img, img, color_combined, thres_img, pers_img]))
         
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= BirdViewTransform()
    obj.run()