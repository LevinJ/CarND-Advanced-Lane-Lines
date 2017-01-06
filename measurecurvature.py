import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from locatelanepixels import LocateLanePixel
import matplotlib.pyplot as plt
from frametracking import g_frame_tracking


class MeasueCurvature(LocateLanePixel):
    def __init__(self):
        LocateLanePixel.__init__(self)
        return
    def fit_lane_lines(self, img,left_pixels, right_pixels, lane_pixel_num):
#         plt.imshow(img[...,::-1])
        if (len(left_pixels) == 0 ) or (len(right_pixels) == 0):
            g_frame_tracking.left_lines.detected = False
            g_frame_tracking.left_lines.detected = False
            print("No lines are detected!!")
            return img,None
        g_frame_tracking.left_lines.detected = True
        g_frame_tracking.right_lines.detected = True
        y_min = min(left_pixels[:,1].min(), right_pixels[:,1].min())
        left_fit, left_fity,left_fitx = self.__fit_lane_line(img, left_pixels,y_min)
        right_fit, right_fity,right_fitx = self.__fit_lane_line(img, right_pixels,y_min)
        g_frame_tracking.left_lines.add_last_fit(left_fit, left_fity,left_fitx)
        g_frame_tracking.right_lines.add_last_fit(right_fit, right_fity,right_fitx)
        

        self.left_fity = left_fity
        self.left_fitx = left_fitx
        self.right_fity = right_fity
        self.right_fitx = right_fitx
        
        fit_pts= self.__concat_fit_pts(left_fity, left_fitx, right_fity, right_fitx)
        img_fitline = img.copy()
        cv2.fillPoly(img_fitline, np.int_([fit_pts]), (0,255, 0))
        self.__cal_curvature( img,  left_fity,left_fitx, right_fity,right_fitx,lane_pixel_num)
        self.__cal_shift_from_center(img, left_fitx, right_fitx, lane_pixel_num)
        return img_fitline,fit_pts
    def __concat_fit_pts(self, left_fity,left_fitx,right_fity,right_fitx):
        pts_left = np.concatenate((left_fitx.reshape(-1,1), left_fity.reshape(-1,1)), axis = 1)
        
        pts_right = np.flipud(np.concatenate((right_fitx.reshape(-1,1) , right_fity.reshape(-1,1)), axis = 1))
        fit_pts = np.concatenate((pts_left, pts_right), axis=0)
        return fit_pts
    def __cal_shift_from_center(self, img,  left_fitx, right_fitx,lane_pixel_num):
        img_width = img.shape[1]
        xm_per_pix = 3.7/float(lane_pixel_num)
        left_bottom_x = left_fitx[-1]
        right_bottom_x = right_fitx[-1]
        lane_center = (left_bottom_x + right_bottom_x)/2.0
        car_center = img_width/2.0
        shift = (car_center - lane_center) * xm_per_pix
        if shift > 0:
            shift_info = 'Vehicle is {:.2f}m right of center'.format(abs(shift))
        else:
            shift_info =  'Vehicle is {:.2f}m left of center'.format(abs(shift))
        print(shift_info)
        
        self.shift = shift
        self.shift_info = shift_info
        return 
    
    def __cal_curvature(self, img,  left_fity,left_fitx, right_fity,right_fitx,lane_pixel_num):
        img_height = img.shape[0]
        y_eval = img_height
        
        ym_per_pix = 30/float(img_height)
        xm_per_pix = 3.7/float(lane_pixel_num)
        
        left_fit_cr = np.polyfit(left_fity*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_fity*ym_per_pix, right_fitx*xm_per_pix, 2)
        
        self.left_curvature  = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
        self.right_curvature  = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
        self.average_curvature = (self.left_curvature  + self.right_curvature )/2.0
        self.curvature_info = "Radius of Curvature is {:.0f}m".format(self.average_curvature)
        print(self.curvature_info)
        
        
        return
    def map_back_road(self, img, fit_pts, Minv):
        if fit_pts is None:
            if g_frame_tracking.use_last_lane_unwrap():
                newwarp = g_frame_tracking.last_unwrap
            else:
                return img
                
        else:
            color_warp = np.zeros_like(img).astype(np.uint8)  
            cv2.fillPoly(color_warp, np.int_([fit_pts]), (0,255, 0))       
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
            g_frame_tracking.add_last_unwrap(newwarp)
            new_roi_img = self.__get_roi_image(newwarp)
            g_frame_tracking.add_last_roi(new_roi_img)

        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,self.curvature_info,(100,50), font, 1,(255,255,255),2)
        cv2.putText(result,self.shift_info,(100,90), font, 1,(255,255,255),2)
        

        return result
    
    def __fit_lane_line(self, img, pixels,y_min):
        img_height = img.shape[0]
        x = pixels[:,0]
        y = pixels[:,1]
        fit = np.polyfit(y, x, 2)
        fity = np.linspace(y_min, img_height, num=10)
        fitx = fit[0]*fity**2 + fit[1]*fity + fit[2]
        return fit, fity,fitx
    def process_image_BGR(self, initial_img):
        original_img, img, color_combined, thres_img  = self.thresh_one_image(initial_img)
        pers_img, Minv,lane_pixel_num = self.bird_view(thres_img)
       
        img_with_windows,img_left_right,left_pixels, right_pixels,hist_img= self.locate_lane_pixels(pers_img)
        img_fitline, fit_pts = self.fit_lane_lines(img_left_right,left_pixels, right_pixels, lane_pixel_num)
        map_back_img = self.map_back_road(img, fit_pts, Minv)
        
        thres_imgs = self.stack_image_horizontal([original_img, img, color_combined, thres_img])
        transformed_imgs = self.stack_image_horizontal([pers_img, img_with_windows,img_left_right,img_fitline])
        right_side = self.stack_image_vertical([thres_imgs,transformed_imgs,hist_img])
        left_side = map_back_img
        final_img = self.stack_image_horizontal([left_side, right_side], max_img_width = left_side.shape[1], max_img_height= left_side.shape[0])
        
        return  final_img

    
    def __get_roi_image(self, img):
        
        temp_gray = img[:,:,1]
        temp_gray[temp_gray<255] = 0
#         plt.imshow(temp_gray, cmap='gray')
        
        start_y = np.argwhere(temp_gray.sum(axis=1) == 0).max()

        left_pts = []
        right_pts = []
        x_gap = 20
        y_gap = 50
        for i in np.linspace(start_y+1, temp_gray.shape[0]-1, num=10):
            i =int(i)
            cur_row = temp_gray[i, :]
            if((cur_row==255).sum() < 10):
                continue
             
            xs = np.argwhere(cur_row==255)
            leftx = xs.min()
            rightx=xs.max()
            left_pts.append((leftx-x_gap,i))
            right_pts.append((rightx+ x_gap, i))
        
        y_min = start_y - y_gap
        left_fity, left_fitx = self.__get_fit_pts(left_pts, temp_gray, y_min)
        right_fity, right_fitx = self.__get_fit_pts(right_pts, temp_gray, y_min)
        fit_pts = self.__concat_fit_pts(left_fity, left_fitx, right_fity, right_fitx)
        color_warp = np.zeros_like(img).astype(np.uint8)
        
        cv2.fillPoly(color_warp, np.int_([fit_pts]), (255,255, 255))
#         plt.imshow(color_warp)
        
            
        
        return color_warp
    def __get_fit_pts(self, pts, temp_gray, y_min):
        pts = np.asarray(pts)
        fit = np.polyfit(pts[:,1], pts[:,0], 2)
        fity = np.linspace(y_min, temp_gray.shape[0], num=10)
        fitx = fit[0]*fity**2 + fit[1]*fity + fit[2]
        return fity,fitx
    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
          './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
#         fnames = ['./test_images/challenge2.jpg']
#         fnames = ['./test_images/test5.jpg']
        

        res_imgs = []
        for fname in fnames:
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