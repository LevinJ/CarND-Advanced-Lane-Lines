
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

# Define a class to receive the characteristics of each line detection
class Lines():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        
        self.last_fitx = None
        self.last_fity = None
        self.last_fit = None
    def add_last_fit(self, fit, fity, fitx):
        self.last_fitx = fitx
        self.last_fity = fity
        self.last_fit = fit
        self.recent_xfitted.append(self.last_fitx)
        return
    
    


class FrameTracking():
    def __init__(self):
        self.enable_frame_tracking = True

        self.left_lines = Lines()
        self.right_lines = Lines()
        self.last_roi = None

        return

    def add_last_roi(self, roi):
        if self.enable_frame_tracking:
            self.last_roi = roi
        return
    def __is_last_frame_confident(self):
        return self.left_lines.detected and self.right_lines.detected

    def use_last_lane_locate_pixels(self):
        return  self.enable_frame_tracking and self.__is_last_frame_confident()
    def use_last_lane_area_as_roi(self):
        return self.enable_frame_tracking and self.__is_last_frame_confident()
 
        
    def run(self):
        
        return


g_frame_tracking= FrameTracking()
if __name__ == "__main__":   
    obj= FrameTracking()
    obj.run()