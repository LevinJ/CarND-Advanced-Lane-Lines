import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip
from findlane import Findlane

class LaneDetection(Findlane):
    
    def __init__(self):
       
        return
    
    def process_image_BGR(self, initial_img):
        original_img, img, thres_img = self.thresh_one_image(initial_img)
        pers_img, _ ,_= self.bird_view(thres_img)
       
        img_with_windows,img_left_right,_,_ = self.locate_lane_pixels(pers_img)
        final_img = self.stack_image_horizontal([original_img, img, thres_img, pers_img, img_with_windows,img_left_right])
        
        
        return  final_img
    def process_image_file_path(self, img_file_path):
        #load the image
        initial_img = self.load_image(img_file_path)
#         plt.imshow(initial_img)
        final_img = self.process_image_BGR(initial_img)

        img_file_name = os.path.basename(img_file_path)
        new_img_file_path = os.path.dirname(img_file_path) + '/' + os.path.splitext(img_file_name)[0] + '_withlane' + os.path.splitext(img_file_name)[1]
        self.save_image(final_img, new_img_file_path)
        print("save to {}".format(new_img_file_path))
       
        plt.imshow(final_img )
#         plt.show()

        
        return final_img
    
    def test_on_one_image(self, img_file_path):
        self.process_image_file_path(img_file_path)
        plt.show()
        return
    def test_on_images(self):
        img_file_paths = ['solidWhiteCurve.jpg',
                             'solidWhiteRight.jpg',
                             'solidYellowCurve.jpg',
                             'solidYellowCurve2.jpg',
                             'solidYellowLeft.jpg',
                             'whiteCarLaneSwitch.jpg']
        img_file_paths = ['../test_images/'+ file_path for file_path in img_file_paths]
        for img_file_path in img_file_paths:
            print("process image file {}".format(img_file_path))
            self.process_image_file_path(img_file_path)
        print("Done with processing images")
#             break
            
        
        return
    def test_on_videos(self, input_video, output_video):
        clip1 = VideoFileClip(input_video)
        white_clip = clip1.fl_image(self.process_image_BGR)
        white_clip.write_videofile(output_video, audio=False)
        return
    def run(self):
#         self.test_on_one_image('../test_images/solidYellowCurve.jpg')
#         self.test_on_images()
        self.test_on_videos('../project_video.mp4','../project.mp4')

        plt.show()
        
        return






if __name__ == "__main__":   
    obj= LaneDetection()
    obj.run()