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
        plt.imshow(img, cmap='gray')
        img_height = img.shape[0]
        num_histogram = 10
        y_step = img_height/num_histogram
        peak_xs = []
        peak_ys= []
        end = img_height
        
        histogram = np.sum(img[int(img_height/2):], axis=0)
        indexes = detect_peaks(histogram, mph=10, mpd=650)
        if (len(indexes) != 2):
            raise Exception('unexpected number of peaks')
        sliding_window_width = 200
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
        
        
        print(indexes)
        
        
        return
    def __locate_true_centers(self, img, peak_y, sliding_windows):
        new_sliding_windows = []
        for sliding_window in sliding_windows:
            new_sliding_windows.extend(self.__locate_true_center(img, peak_y, sliding_window))
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
        sliding_windows = [[indexes-sliding_window_width/2, indexes+ sliding_window_width/2]
        if sliding_windows[0] < 0:
            sliding_windows[0] = 0
            print('end reached, left side')
        if sliding_windows[1] > img.shape[1]:
            sliding_windows[1] = img.shape[1]
            print('end reached, right side')
            
        return sliding_windows

    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
#         fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
#                   './test_images/test5.jpg','./test_images/test6.jpg']
        fnames = ['./test_images/test2.jpg']

        res_imgs = []
        for fname in fnames:
            original_img, img, thres_img = self.thresh_one_image(fname)
            pers_img = self.bird_view(thres_img)
            self.locate_lane_pixels(pers_img)
            res_imgs.append(self.stack_image_horizontal([original_img, img, thres_img, pers_img]))
        
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= Findlane()
    obj.run()