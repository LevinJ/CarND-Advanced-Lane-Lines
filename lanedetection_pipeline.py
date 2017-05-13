import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip
from measurecurvature import MeasueCurvature

class LaneDetection(MeasueCurvature):
    
    def __init__(self):
        MeasueCurvature.__init__(self)
        self.count = 0
        self.debug_frame = False
        self.debug_frame_id = 1178
       
        return
    
    def process_image(self, initial_img):
        try:
            if self.debug_frame and (self.debug_frame_id != self.count):
                self.count = self.count + 1
                return initial_img
            img_BGR = initial_img[...,::-1]
            
            print('frame {}'.format(self.count))
             
            final_img = self.process_image_BGR(img_BGR)
            final_img = final_img[...,::-1]
            
            self.count = self.count + 1
            plt.imshow(final_img)
        except:
            plt.imsave('exception_img.jpg', initial_img)
            raise 
        return  final_img

    
    def test_on_one_image(self, img_file_path):
        #load the image
        initial_img = cv2.imread(img_file_path)
#         plt.imshow(initial_img)
        final_img = self.process_image_BGR(initial_img)

        plt.imshow(final_img )

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
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(output_video, audio=False)
        return
    def test_on_frame(self):
        clip = VideoFileClip('./project_video.mp4')
        initial_img = None
        for img in clip.iter_frames():
            if self.count == 0:
                initial_img = img
                plt.imsave('./test_images/img_0.jpg', initial_img)
                break
            else:
                self.count = self.count + 1
            
            
          
        img_BGR = initial_img[...,::-1]
          
        final_img = self.process_image_BGR(img_BGR)
        
        return final_img
    def run(self):
#         self.test_on_one_image('excpetion_img.jpg')
        self.test_on_frame()
#         self.test_on_videos('./project_video.mp4','./project.mp4')
#         self.test_on_videos('./challenge_video.mp4','./challenge.mp4')
#         self.test_on_videos('./harder_challenge_video.mp4','./harder_challenge.mp4')

#         plt.show()
        
        return






if __name__ == "__main__":   
    obj= LaneDetection()
    obj.run()