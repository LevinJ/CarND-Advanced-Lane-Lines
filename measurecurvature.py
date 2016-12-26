import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from findlane import Findlane
import matplotlib.pyplot as plt


class MeasueCurvature(Findlane):
    def __init__(self):
        Findlane.__init__(self)
        return
    def fit_lane_lines(self, img,left_pixels, right_pixels):
        
        left_fit, left_fity,left_fitx = self.__fit_lane_line(img, left_pixels)
        right_fit, right_fity,right_fitx = self.__fit_lane_line(img, right_pixels)
        pts_left = np.concatenate((left_fitx.reshape(-1,1), left_fity.reshape(-1,1)), axis = 1)
        
        pts_right = np.flipud(np.concatenate((right_fitx.reshape(-1,1) , right_fity.reshape(-1,1)), axis = 1))
        fit_pts = np.concatenate((pts_left, pts_right), axis=0)
        img_fitline = img.copy()
        cv2.fillPoly(img_fitline, np.int_([fit_pts]), (0,255, 0))
        return img_fitline,fit_pts
    def map_back_road(self, img, fit_pts, Minv):
        color_warp = np.zeros_like(img).astype(np.uint8)
        
        cv2.fillPoly(color_warp, np.int_([fit_pts]), (0,255, 0))
        
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        return result
    
    def __fit_lane_line(self, img, pixels):
        img_height = img.shape[0]
        x = pixels[:,0]
        y = pixels[:,1]
        fit = np.polyfit(y, x, 2)
        fity = np.linspace(y.min(), img_height, num=10)
        fitx = fit[0]*fity**2 + fit[1]*fity + fit[2]
        return fit, fity,fitx

    
    def run(self):
        fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
                  './test_images/straight16.jpg','./test_images/straight17.jpg']
#         fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
#                   './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/test2.jpg']

        res_imgs = []
        for fname in fnames:
            original_img, img, thres_img = self.thresh_one_image(fname)
            pers_img, Minv = self.bird_view(thres_img)
           
            img_with_windows,img_left_right,left_pixels, right_pixels= self.locate_lane_pixels(pers_img)
            img_fitline, fit_pts = self.fit_lane_lines(img_left_right,left_pixels, right_pixels)
            map_back_img = self.map_back_road(img, fit_pts, Minv)
            res_imgs.append(self.stack_image_horizontal([original_img, img, thres_img, pers_img, img_with_windows,img_left_right,img_fitline, map_back_img]))
        
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= MeasueCurvature()
    obj.run()