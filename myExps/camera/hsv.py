import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #orange
    #color_min = np.array([5, 50, 50],np.uint8)
    #color_max = np.array([15, 255, 255],np.uint8)

    #blue
    #color_min = np.array([110,50,50])
    #color_max = np.array([130,255,255])

    #green bgr [0,255,0]; 60
    #color_min = np.array([50, 50, 50])
    #color_max = np.array([70, 255, 255])

    #pink bgr [203,192,255]; 175
    color_min = np.array([150, 100, 150])
    color_max = np.array([250, 255, 255])

    #color_min = np.array([150, 120, 150])
    #color_max = np.array([250, 255, 255])

    #define GREENTHRESH 50, 130, 80, 90, 256, 256
#define ORANGETHRESH 10, 150, 200, 20, 256, 256
#define YELLOWTHRESH 160, 90, 140, 180, 256, 256

    #MY GREEN:
    #color_min = np.array([50, 130, 80])
    #color_max = np.array([90, 255, 255])

    #color_min = np.array([50, 130, 80])
    #color_max = np.array([90, 255, 255])

    #MY ORANGE:
    #color_min = np.array([10, 150, 200])
    #color_max = np.array([20, 255, 255])

    #color_min = np.array([5, 150, 200])
    #color_max = np.array([20, 255, 255])

    #color_min = np.array([0, 100, 100])
    #color_max = np.array([90, 255, 255])

    #color_min = np.array([0, 0, 0])
    #color_max = np.array([45, 255, 255])

    #color_min = np.array([15, 60, 245])
    #color_max = np.array([30, 100, 255])

    #MY YELLOW:
    #color_min = np.array([25, 100, 150])
    #color_max = np.array([45, 255, 255])

    #color_min = np.array([25, 70, 150])
    #color_max = np.array([45, 255, 255])

    frame_threshed = cv2.inRange(hsv, color_min, color_max)

    M = cv2.moments(frame_threshed)
    centroid_x = int(M['m10']/(M['m00']+1))
    centroid_y = int(M['m01']/(M['m00']+1))
    cv2.circle(frame_threshed, (centroid_x, centroid_y), 3, 60, 1)

    cv2.line(frame,(1,240),(640,240),(255,0,0),1)
    cv2.line(frame,(320,1),(320,480),(255,0,0),1)

    # Display the resulting frame
    #cv2.imshow('frame',hsv)
    cv2.imshow('fram',frame)
    cv2.imshow('HSV thresholded frame',frame_threshed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#finding colors
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
