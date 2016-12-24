import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from transform import Transform
from scipy import signal
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks


class Findlane(Transform):
    def __init__(self):
        Transform.__init__(self)
        return
    def locate_lane_pixels(self, img):
#         plt.imshow(img, cmap='gray')
        img_height = img.shape[0]
        num_histogram = 5
        y_step = img_height/num_histogram
        peak_xs = []
        peak_ys= []
        end = img_height
        
        histogram = np.sum(img[int(img_height/2):], axis=0)
        indexes = detect_peaks(histogram, mph=10, mpd=650)
        if (len(indexes) != 2):
            raise Exception('unexpected number of peaks')
        sliding_window_width = 300
        sliding_windows = [(indexes[0]-sliding_window_width/2, indexes[0]+ sliding_window_width/2),
                           (indexes[1]-sliding_window_width/2, indexes[1]+ sliding_window_width/2)]
        while end > 0:
            start = end - y_step
            if start < 0:
                start = 0
            peak_ys.append((start, end))
            sliding_windows = self.__locate_true_centers(img, (start, end), sliding_windows)
            peak_xs.append(sliding_windows)
            end = end - y_step
        
        
        
        img_with_windows = img.copy()
        self.__draw_sliding_windows(img_with_windows, peak_ys, peak_xs)
#         plt.imshow(img, cmap='gray')
        
        
        return img_with_windows
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
            new_sliding_windows.append(self.__locate_true_center(img, peak_y, sliding_window))
        return new_sliding_windows
    def __locate_true_center(self, img, peak_y, sliding_window):
        if sliding_window is None:
            return None
        sliding_window_width = sliding_window[1] - sliding_window[0]
        histogram = np.sum(img[int(peak_y[0]):int(peak_y[1]), int(sliding_window[0]):int(sliding_window[1])], axis=0)

        indexes = detect_peaks(histogram, mph=1, mpd=650)
        if len(indexes) != 1:
            print('unexpected peak number')
            return None
        indexes = sliding_window[0] + indexes[0]
        sliding_windows = [indexes-sliding_window_width/2, indexes+ sliding_window_width/2]
        if sliding_windows[0] < 0:
            sliding_windows[0] = 0
            print('end reached, left side')
        if sliding_windows[1] > img.shape[1]:
            sliding_windows[1] = img.shape[1]
            print('end reached, right side')
            
        return sliding_windows
    def __show_rectangel(self,img):
        plt.imshow(img, cmap='gray')
        pt1 = (229, 648)
        pt2 =(429, 720)
        cv2.rectangle(img, pt1, pt2, 1,thickness=5)
        plt.imshow(img, cmap='gray')
        return

    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
                  './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/test2.jpg']

        res_imgs = []
        for fname in fnames:
            original_img, img, thres_img = self.thresh_one_image(fname)
            pers_img = self.bird_view(thres_img)
           
            img_with_windows = self.locate_lane_pixels(pers_img)
            res_imgs.append(self.stack_image_horizontal([original_img, img, thres_img, pers_img, img_with_windows]))
        
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= Findlane()
    obj.run()