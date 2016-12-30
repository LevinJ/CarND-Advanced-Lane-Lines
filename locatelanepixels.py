import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from birdview import BirdViewTransform
from scipy import signal
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks


class LocateLanePixel(BirdViewTransform):
    def __init__(self):
        BirdViewTransform.__init__(self)
        return
    def locate_lane_pixels(self, img):
#         plt.imshow(img, cmap='gray')
        img_height = img.shape[0]
        num_histogram = 5
        y_step = int(img_height/num_histogram)
        
        peak_ys= []
        end = img_height
        self.img_width = img.shape[1]
        self.peak_xs = []
        
        histogram = np.sum(img[int(img_height/2):], axis=0)
        indexes = detect_peaks(histogram, mph=10, mpd=650)
        if (len(indexes) != 2):
            print("unexpected number of peaks!!!")
            return img, img,[], []
        sliding_window_width = 160
        sliding_windows = [(indexes[0]-int(sliding_window_width/2), indexes[0]+ int(sliding_window_width/2)),
                           (indexes[1]-int(sliding_window_width/2), indexes[1]+ int(sliding_window_width/2))]
        while end > 0:
            start = end - y_step
            if start < 0:
                start = 0
            peak_ys.append((start, end))
            sliding_windows = self.__locate_true_centers(img, (start, end), self.__get_initial_sliding_windows(sliding_windows, img))
            self.peak_xs.append(sliding_windows)
            end = end - y_step
        
        
        
        img_with_windows = img.copy()
        self.__draw_sliding_windows(img_with_windows, peak_ys, self.peak_xs)
        left_pixels, right_pixels = self.__identify_lane_pixles(img, peak_ys, self.peak_xs)
        img_left_right = self.__draw_left_right_pixels(img, left_pixels, right_pixels)
        return img_with_windows, img_left_right,left_pixels, right_pixels
    def __get_initial_sliding_windows(self, sliding_windows, img):
        if len(self.peak_xs) <= 1:
            #if this is the first sliding windows, then just return
            return sliding_windows
        new_sliding_windows = []
        for i in range(len(sliding_windows)):
            new_sliding_window = self.__get_intial_sliding_window(sliding_windows[i], np.array(self.peak_xs)[:,i], img)
            new_sliding_windows.append(new_sliding_window)
        
        return  new_sliding_windows
   
    def __get_intial_sliding_window(self, sliding_window, peak_xs, img):
        peak_xs_list = peak_xs.tolist()
        if None in peak_xs_list:
            # do not detect window if detection once faled at the lower part of the image
            return sliding_window
        peak_xs = np.asarray(peak_xs_list)
        xs = np.array(list(peak_xs[:,0]))
        shiftxs = xs[1:]-xs[0:-1]
        #when the shfit is small, hold our judegement about the direction of the lane
        shiftxs[(shiftxs< 3) & (shiftxs>-3)] = 0
        
        sliding_window = [sliding_window[0] + shiftxs[-1], sliding_window[1]+ shiftxs[-1]]
        if sliding_window[0] < 0:
            sliding_window[0] = 0
#             print('end reached, left side')
        if sliding_window[1] > img.shape[1]:
            sliding_window[1] = img.shape[1]
       
        return sliding_window
    def __draw_left_right_pixels(self,img, left_pixels, right_pixels):
        
        zero = np.zeros_like(img).astype(np.uint8)
        left = np.zeros_like(img).astype(np.uint8)
        right = np.zeros_like(img).astype(np.uint8)
        if len(left_pixels) != 0:
            left[left_pixels[:,1], left_pixels[:,0]] = 255
        if len(right_pixels) !=0 :    
            right[right_pixels[:,1], right_pixels[:,0]] = 255
       
        img_left_right = np.dstack((left,zero,right))
        return img_left_right
    def __identify_lane_pixles(self, img, peak_ys, peak_xs):
        left_pixels = []
        right_pixels =[]
        for i in range(len(peak_ys)):
            peak_y = peak_ys[i]
            y1 = peak_y[0]
            y2 = peak_y[1]
            left_x,right_x = peak_xs[i]
            if left_x is not None:
                x1,x2 = left_x
                y_relative,x_relative = img[y1:y2,x1:x2].nonzero()
                y = (y_relative + y1).reshape(-1,1)
                x = (x_relative + x1).reshape(-1,1)
                pixels = np.concatenate([x,y], axis = 1)
                left_pixels.extend(pixels) 
            if right_x is not None:
                x1,x2 = right_x
                y_relative,x_relative = img[y1:y2,x1:x2].nonzero()
                y = (y_relative + y1).reshape(-1,1)
                x = (x_relative + x1).reshape(-1,1)
                pixels = np.concatenate([x,y], axis = 1)
                right_pixels.extend(pixels) 
            
        left_pixels = np.asarray(left_pixels).astype(np.int32)
        right_pixels = np.asarray(right_pixels).astype(np.int32)
        return  left_pixels, right_pixels
  
    def __draw_sliding_windows(self,img, peak_ys, peak_xs):
        sliding_windows_pts =[]
        for i in range(len(peak_ys)):
            y1, y2 = peak_ys[i]
            leftx, rightx = peak_xs[i]
            if leftx is not None:
                leftx1, leftx2= leftx
                left_window = [(leftx1, y1),(leftx2, y1),(leftx2, y2),(leftx1, y2)]
                left_window = np.asarray(left_window).astype(np.int32)
                sliding_windows_pts.append(left_window)
                cv2.rectangle(img, tuple(left_window[0]), tuple(left_window[2]),  1,thickness=5)
                
            
            if rightx is not None:
                rightx1,rightx2 = rightx
                right_window = [(rightx1, y1),(rightx2, y1),(rightx2, y2),(rightx1, y2)]
                right_window = np.asarray(right_window).astype(np.int32)
                sliding_windows_pts.append(right_window)
                cv2.rectangle(img, tuple(right_window[0]), tuple(right_window[2]),  1,thickness=5)
            
        
            
        
        return
    def __locate_true_centers(self, img, peak_y, sliding_windows):
        new_sliding_windows = []
        for sliding_window in sliding_windows:
            new_window = self.__locate_true_center(img, peak_y, sliding_window)
            for _ in range(3):
                if new_window is not None:
                    res_window = self.__locate_true_center(img, peak_y, new_window)
                    if sum(res_window) == sum(new_window):
                        #if the slidig windows has no adjustment, then no longer fine tune
                        break
                    new_window = res_window 
            new_sliding_windows.append(new_window)
        return self.__adjust_sliding_windows(new_sliding_windows)
    def __adjust_sliding_windows(self, sliding_windows):
        
        if len(self.peak_xs) <= 1:
            #if this is the first sliding windows, then just return
            return sliding_windows
#         
        new_sliding_windows = []
         
         
        for i in range(len(sliding_windows)):
            new_sliding_window = self.__adjust_sliding_window(sliding_windows[i], np.array(self.peak_xs)[:,i])
            new_sliding_windows.append(new_sliding_window)
            
        return new_sliding_windows
    def __is_same_sign(self, x1, x2):
        if x1==0 or x2==0:
            return True
        if x1 > 0 and x2 > 0:
            return True
        if x1<0 and x2<0:
            return True
        return False
    def __adjust_sliding_window(self, sliding_window, peak_xs):
        if sliding_window is None:
            return None
        if (sliding_window[0]<=0) or (sliding_window[1]>=self.img_width):
            return sliding_window
        peak_xs_list = peak_xs.tolist()
        if None in peak_xs_list:
            # do not detect window if detection once faled at the lower part of the image
            return None
        peak_xs = np.asarray(peak_xs_list)
        try:
            xs = np.array(list(peak_xs[:,0]) + [sliding_window[0]])
        except:
            raise Exception('error')
        shiftxs = xs[1:]-xs[0:-1]
        #when the shfit is small, hold our judegement about the direction of the lane
        shiftxs[(shiftxs< 3) & (shiftxs>-3)] = 0
        
#         print("shift {}".format(shiftxs))
        
        if(not self.__is_same_sign(shiftxs[-1], shiftxs[-2])):
            sliding_window = (peak_xs[-1] + shiftxs[-2]).tolist()
            print('adjust sliding windows')
        
        
        return sliding_window
    def __locate_true_center(self, img, peak_y, sliding_window):
        if sliding_window is None:
            return None
        sliding_window_width = sliding_window[1] - sliding_window[0]
        histogram = np.sum(img[int(peak_y[0]):int(peak_y[1]), int(sliding_window[0]):int(sliding_window[1])], axis=0)
        if histogram.sum()==0:
            return None
        try:
            index = np.sum((histogram/float(histogram.sum())) * (np.arange(len(histogram)) + 1))
            index = int(index -1)
        except:
            raise Exception('exception in extracting center')
        

#         indexes = detect_peaks(histogram, mph=1, mpd=650)
#         if len(indexes) != 1:
#             print('unexpected peak number')
#             return None
        indexes = sliding_window[0] + index
        sliding_windows = [indexes-int(sliding_window_width/2), indexes+ int(sliding_window_width/2)]
        if sliding_windows[0] < 0:
            sliding_windows[0] = 0
#             print('end reached, left side')
        if sliding_windows[1] > img.shape[1]:
            sliding_windows[1] = img.shape[1]
#             print('end reached, right side')
            
        return sliding_windows
    def __show_rectangel(self,img):
        plt.imshow(img, cmap='gray')
        pt1 = (229, 648)
        pt2 =(429, 720)
        cv2.rectangle(img, pt1, pt2, 1,thickness=5)
        plt.imshow(img, cmap='gray')
        return

    
    def run(self):
        fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
                  './test_images/straight16.jpg','./test_images/straight17.jpg']
#         fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
#                   './test_images/test5.jpg','./test_images/test6.jpg','./exception_img.jpg']
#         fnames = ['./exception_img.jpg']
#         fnames = ['./test_images/test5.jpg']

        res_imgs = []
        for fname in fnames:
            original_img, img, color_combined, thres_img  = self.thresh_one_image_fname(fname)
            pers_img, _ ,_= self.bird_view(thres_img)
           
            img_with_windows,img_left_right,_,_ = self.locate_lane_pixels(pers_img)
            res_imgs.append(self.stack_image_horizontal([original_img, img, color_combined, thres_img, pers_img, img_with_windows,img_left_right]))
        
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= LocateLanePixel()
    obj.run()