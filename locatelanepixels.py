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
from frametracking import FrameTracking, g_frame_tracking


class LocateLanePixel(BirdViewTransform):
    def __init__(self):
        BirdViewTransform.__init__(self)
        return
    def locate_lane_pixels(self, img):
#         plt.imshow(img, cmap='gray')
        img_height = img.shape[0]
        histogram = np.sum(img[int(img_height/2):], axis=0)
        
        if g_frame_tracking.use_last_lane_locate_pixels():
            found_peaks = (int(g_frame_tracking.left_lines.last_fitx[-1]), int(g_frame_tracking.right_lines.last_fitx[-1]))
            sliding_window_width = 160
            prefix_str="Last Frame Center"

        else:    
            found_peaks = self.__find_root_peaks(histogram)
            sliding_window_width = 160
            prefix_str=""
        
        self.hist_img = self.__get_histogram_img(img, histogram, found_peaks,prefix_str=prefix_str)
#         plt.imshow(self.hist_img)
        if (len(found_peaks) != 2):
            print("unexpected number of peaks!!!")
            return img, img,[], [],self.hist_img
                
                
        return self.__locate_lane_pixels_with_root_peaks(img,found_peaks,self.hist_img,sliding_window_width)
    def __find_root_peaks(self, histogram):
        found_peaks = detect_peaks(histogram, mph=10, mpd=650)
        if len(found_peaks)!=2:
            found_peaks = np.array([300,920])
        return found_peaks
    def __locate_lane_pixels_with_root_peaks(self, img,found_peaks,hist_img, sliding_window_width):
        img_height = img.shape[0]
        num_histogram = 5
        y_step = int(img_height/num_histogram)
        
        self.peak_ys= []
        end = img_height
        self.img_width = img.shape[1]
        self.peak_xs = []
        sliding_windows = [(found_peaks[0]-int(sliding_window_width/2), found_peaks[0]+ int(sliding_window_width/2)),
                           (found_peaks[1]-int(sliding_window_width/2), found_peaks[1]+ int(sliding_window_width/2))]
        while end > 0:
            start = end - y_step
            if start < 0:
                start = 0
            self.peak_ys.append((start, end))
            sliding_windows = self.__fine_tune_sliding_windows(img, (start, end), self.__get_initial_sliding_windows(sliding_windows, img))
            self.peak_xs.append(sliding_windows)
            end = end - y_step
        
        
        
        img_with_windows = img.copy()
        self.__draw_sliding_windows(img_with_windows, self.peak_ys, self.peak_xs)
        left_pixels, right_pixels = self.__identify_lane_pixles(img, self.peak_ys, self.peak_xs)
        img_left_right = self.__draw_left_right_pixels(img, left_pixels, right_pixels)
#         plt.imshow(hist_img)
        
        return img_with_windows, img_left_right,left_pixels, right_pixels,hist_img
    def __get_histogram_img(self, img, histogram, found_peaks, prefix_str=""):
        img_height = img.shape[0]
        hist_img = img.copy()
        hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
        hist_img = np.uint8(255*hist_img/np.max(hist_img))
        width = len(histogram)
        ys = -histogram[:, np.newaxis] + img_height
        xs = np.arange(0, width)[:, np.newaxis]
        pts = np.concatenate([xs,ys], axis = 1).astype(np.int32)
        cv2.polylines(hist_img, [pts], color=(255,0,0),isClosed=False,thickness=6)
        if len(found_peaks) != 0:
            peaks_info = prefix_str + '{}'.format(found_peaks)
            cv2.putText(hist_img,peaks_info,(100,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),4)
        for peak in found_peaks:
            pt1 = (peak, img_height)
            pt2 = (peak, int(img_height/2))
            cv2.line(hist_img, pt1,pt2,(0,0,255),thickness=5)
        
        return hist_img
    def __get_initial_sliding_windows(self, sliding_windows, img):
        new_sliding_windows = []
        peak_ys = self.peak_ys[-1]
        
        if len(self.peak_xs) <= 1:
            #if this is the first sliding windows, then just return
            new_sliding_windows = sliding_windows
        
        elif g_frame_tracking.use_last_lane_locate_pixels():
            windows_width = 160
            current_index = -len(self.peak_ys)
            if (g_frame_tracking.left_lines.last_fitx[current_index] is not None) and (g_frame_tracking.right_lines.last_fitx[current_index] is not None):
                left_window = [g_frame_tracking.left_lines.last_fitx[current_index]-int(windows_width/2), 
                                g_frame_tracking.left_lines.last_fitx[current_index]+int(windows_width/2)]
                right_window = [g_frame_tracking.right_lines.last_fitx[current_index]-int(windows_width/2), 
                                g_frame_tracking.right_lines.last_fitx[current_index]+int(windows_width/2)]
                new_sliding_windows = [left_window, right_window]
                
        else:
        
            for i in range(len(sliding_windows)):
                new_sliding_window = self.__get_intial_sliding_window(sliding_windows[i], np.array(self.peak_xs)[:,i], img)
                new_sliding_windows.append(new_sliding_window)
        
        self.__draw_sliding_windows(self.hist_img, [peak_ys], [new_sliding_windows])
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
        #discard some very small or invalid sliding windows
        if sliding_window[1]- sliding_window[0] <= 10:
            return None
        
        
       
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
                if (len(img.shape) == 2):
                    cv2.rectangle(img, tuple(left_window[0]), tuple(left_window[2]),  1,thickness=5)
                else:
                    cv2.rectangle(img, tuple(left_window[0]), tuple(left_window[2]),  (255,255,255),thickness=5)
                
            
            if rightx is not None:
                rightx1,rightx2 = rightx
                right_window = [(rightx1, y1),(rightx2, y1),(rightx2, y2),(rightx1, y2)]
                right_window = np.asarray(right_window).astype(np.int32)
                sliding_windows_pts.append(right_window)
                if (len(img.shape) == 2):
                    cv2.rectangle(img, tuple(right_window[0]), tuple(right_window[2]),  1,thickness=5)
                else:
                    cv2.rectangle(img, tuple(right_window[0]), tuple(right_window[2]),  (255,255,255),thickness=5)
            
        
            
        
        return
    def __fine_tune_sliding_windows(self, img, peak_y, sliding_windows):
        new_sliding_windows = []
        for sliding_window in sliding_windows:
            new_window = self.__fine_tune_sliding_window(img, peak_y, sliding_window)
            for _ in range(3):
                if new_window is not None:
                    res_window = self.__fine_tune_sliding_window(img, peak_y, new_window)
                    try:
                        if sum(res_window) == sum(new_window):
                            #if the slidig windows has no adjustment, then no longer fine tune
                            break
                    except:
                        raise
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
    def __fine_tune_sliding_window(self, img, peak_y, sliding_window):
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
        


        indexes = sliding_window[0] + index
        sliding_windows = [indexes-int(sliding_window_width/2), indexes+ int(sliding_window_width/2)]
        if sliding_windows[0] < 0:
            sliding_windows[0] = 0
#             print('end reached, left side')
        if sliding_windows[1] > img.shape[1]:
            sliding_windows[1] = img.shape[1]
        
        if sliding_windows[0] >= sliding_windows[1]:
            return sliding_window
#             print('end reached, right side')
            
        return sliding_windows
    def __show_rectangel(self,img):
#         plt.imshow(img, cmap='gray')
        pt1 = (229, 648)
        pt2 =(429, 720)
        cv2.rectangle(img, pt1, pt2, 1,thickness=5)
#         plt.imshow(img, cmap='gray')
        return

    
    def run(self):
        fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
                  './test_images/straight16.jpg','./test_images/straight17.jpg']
#         fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
#                   './test_images/test5.jpg','./test_images/test6.jpg','./exception_img.jpg']
        fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
          './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
        fnames = ['./test_images/challenge2.jpg']
#         fnames = ['./test_images/test5.jpg']

        res_imgs = []
        for fname in fnames:
            original_img, img, color_combined, thres_img  = self.thresh_one_image_fname(fname)
            pers_img, _ ,_= self.bird_view(thres_img)
           
            img_with_windows,img_left_right,_,_ ,hist_img= self.locate_lane_pixels(pers_img)
            res_imgs.append(self.stack_image_horizontal([original_img, img, color_combined, thres_img, pers_img, hist_img,img_with_windows,img_left_right]))
        
        res_imgs = np.array(res_imgs)
        res_imgs = np.concatenate(res_imgs, axis=0)
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= LocateLanePixel()
    obj.run()