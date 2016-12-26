import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utility.dumpload import DumpLoad
import glob


class Calibrarte(object):
    def __init__(self):
        return
    def __find_coners(self):
        # prepare object points
        nx = 9#TODO: enter the number of inside corners in x
        ny = 5#TODO: enter the number of inside corners in y
        
        # Make a list of calibration images
        fname = './camera_cal/calibration1.jpg'  #9*5
        img = cv2.imread(fname)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            return
        else:
            print('no corners found!!')
            
    def __calibrate(self):
        dumpload = DumpLoad('./camera_cal/cameracalibration.p')
        if dumpload.isExisiting():
            return dumpload.load()
            
        objpoints = []
        imgpoints = []
        nx = 9
        ny = 6
        objp = np.zeros((nx*ny, 3), np.float32)
        #prepare objpoints, like (0,0,0), (1,0,0), (2,0,0), ... (8,5,0) etc
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        images = glob.glob('./camera_cal/calibration*.jpg')
        for fname in images:
            img = cv2.imread(fname)
        
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            if 'calibration1.jpg' in fname:
                ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)
            else:
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                imgpoints.append(corners)
                if 'calibration1.jpg' in fname:
                    objpoints.append(objp[0:45,:])
                else:
                    objpoints.append(objp)


            else:
                print('no corners found in image {}'.format(fname))
                
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        
        dumpload.dump((mtx, dist))
            
        return mtx, dist
    def undistort(self, img):
        mtx, dist= self.__calibrate()
        
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        return undist_img
    def run(self):
#         fname = './test_images/test1.jpg'
        fname = './camera_cal/calibration1.jpg'
        img = cv2.imread(fname)
        img = img[...,::-1] #convert from opencv bgr to standard rgb
        
        undist = self.undistort(img)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)

        plt.show()
        return



if __name__ == "__main__":   
    obj= Calibrarte()
    obj.run()