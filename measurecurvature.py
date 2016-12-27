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
    def fit_lane_lines(self, img,left_pixels, right_pixels, lane_pixel_num):
        
        _, left_fity,left_fitx = self.__fit_lane_line(img, left_pixels)
        _, right_fity,right_fitx = self.__fit_lane_line(img, right_pixels)
        pts_left = np.concatenate((left_fitx.reshape(-1,1), left_fity.reshape(-1,1)), axis = 1)
        
        pts_right = np.flipud(np.concatenate((right_fitx.reshape(-1,1) , right_fity.reshape(-1,1)), axis = 1))
        fit_pts = np.concatenate((pts_left, pts_right), axis=0)
        img_fitline = img.copy()
        cv2.fillPoly(img_fitline, np.int_([fit_pts]), (0,255, 0))
        self.__cal_curvature( img,  left_fity,left_fitx, right_fity,right_fitx,lane_pixel_num)
        return img_fitline,fit_pts
    def __cal_curvature(self, img,  left_fity,left_fitx, right_fity,right_fitx,lane_pixel_num):
        img_height = img.shape[0]
        y_eval = img_height
        
        ym_per_pix = 30/float(img_height)
        xm_per_pix = 3.7/float(lane_pixel_num)
        
        left_fit_cr = np.polyfit(left_fity*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_fity*ym_per_pix, right_fitx*xm_per_pix, 2)
        
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
        print(left_curverad, 'm', right_curverad, 'm')
        
        
        return
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
    def process_image_BGR(self, initial_img):
        original_img, img, thres_img = self.thresh_one_image(initial_img)
        pers_img, Minv,lane_pixel_num = self.bird_view(thres_img)
       
        img_with_windows,img_left_right,left_pixels, right_pixels= self.locate_lane_pixels(pers_img)
        img_fitline, fit_pts = self.fit_lane_lines(img_left_right,left_pixels, right_pixels, lane_pixel_num)
        map_back_img = self.map_back_road(img, fit_pts, Minv)
        
        thres_imgs = self.stack_image_horizontal([original_img, img, thres_img])
        transformed_imgs = self.stack_image_horizontal([pers_img, img_with_windows,img_left_right,img_fitline])
        right_side = self.stack_image_vertical([thres_imgs,transformed_imgs])
        left_side = map_back_img
        final_img = self.stack_image_horizontal([left_side, right_side])
        
        
        return  final_img
    def process_image_fname(self, fname):
     
        original_img, img, thres_img = self.thresh_one_image_fname(fname)
        pers_img, Minv,lane_pixel_num = self.bird_view(thres_img)
       
        img_with_windows,img_left_right,left_pixels, right_pixels= self.locate_lane_pixels(pers_img)
        img_fitline, fit_pts = self.fit_lane_lines(img_left_right,left_pixels, right_pixels, lane_pixel_num)
        map_back_img = self.map_back_road(img, fit_pts, Minv)
     
        return self.stack_image_horizontal([original_img, img, thres_img, pers_img, img_with_windows,img_left_right,img_fitline, map_back_img])

    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
#         fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
#                   './test_images/test5.jpg','./test_images/test6.jpg']
        fnames = ['./test_images/test2.jpg']

        res_imgs = []
        for fname in fnames:
#             res_img = self.process_image_fname(fname)
            img = cv2.imread(fname)
            res_img = self.process_image_BGR(img)
            res_imgs.append(res_img)
        
        res_imgs = self.stack_image_vertical(res_imgs)

        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= MeasueCurvature()
    obj.run()