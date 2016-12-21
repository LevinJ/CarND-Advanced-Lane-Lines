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
        img_height = img.shape[0]
        histogram = np.sum(img[img_height/2:,:], axis=0)
        plt.plot(histogram)
        indexes = detect_peaks(histogram, mph=10, mpd=500)
        return

    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg',
                  './test_images/test5.jpg','./test_images/test6.jpg']

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