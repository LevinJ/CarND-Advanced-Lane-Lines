import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class Calibrarte(object):
    def __init__(self):
        return
    def find_coners(self):
        # prepare object points
        nx = 9#TODO: enter the number of inside corners in x
        ny = 6#TODO: enter the number of inside corners in y
        
        # Make a list of calibration images
        fname = './camera_cal/calibration1.jpg'
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
    def calibrate(self):
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
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                imgpoints.append(corners)
                objpoints.append(objp)
#                 cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#                 plt.imshow(img)

            else:
                print('no corners found in image {}'.format(fname))
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            
        return ret, mtx, dist, rvecs, tvecs
    def run(self):
#         self.find_coners()
        self.calibrate()
        
        plt.show()
        return

class Undistort(object):
    def __init__(self):

        return
   
    
    def run(self):
        fname = './camera_cal/calibration2.jpg'
        img = cv2.imread(fname)
        cali = Calibrarte()
        ret, mtx, dist, rvecs, tvecs = cali.calibrate()
        
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)

        plt.show()

        return
    


if __name__ == "__main__":   
    obj= Undistort()
    obj.run()