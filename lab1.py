import cv2
import numpy as np
cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))
road_frame = np.zeros((280,480))
upper_right = (195,220)
upper_left = (280,220)
lower_left = (0, 280)
lower_right = (480,280)
coordinates = np.array([upper_left,upper_right, lower_left, lower_right], dtype = np.int32)
newFrame = cv2.fillConvexPoly(road_frame, coordinates, (255,255,255))
winWhiteTrapez = "White Trapezoid"
cv2.namedWindow(winWhiteTrapez)
cv2.moveWindow(winWhiteTrapez,2*480+2,0) 


while True : 
    ret, frame = cam.read()
    winResized = 'Resized'
    cv2.namedWindow(winResized)
    cv2.moveWindow(winResized,0,0)
    winGrayscale = 'GrayScale'
    cv2.namedWindow(winGrayscale)
    cv2.moveWindow(winGrayscale,481,0)
    winStreet = 'JustTheStreet'
    cv2.namedWindow(winStreet)
    cv2.moveWindow(winStreet,3*480+3,0)
    winStretched = 'StretchedStreet'
    cv2.namedWindow(winStretched)
    cv2.moveWindow(winStretched,0,311)
    if ret is False : 
        break
    else :
        frameResized = cv2.resize(frame, (480, 280))
        frameGrayscale = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
        #out.write(frame)
    cv2.imshow(winResized, frameResized)
    cv2.imshow(winGrayscale, frameGrayscale)
    cv2.imshow(winWhiteTrapez, newFrame * 255)
    frameStreet = frameGrayscale * (newFrame * 255)
    cv2.imshow(winStreet, frameStreet)
    cv2.imshow(winStretched,frameGrayscale)
    coordinates = np.float32(coordinates)
    magic_matrix = cv2.getPerspectiveTransform(coordinates, np.float32([(480,0), (0,0),(0,280),(480,280)]))
    frameStretched =cv2.warpPerspective(frameStreet, magic_matrix,(480,280))
    cv2.imshow(winStretched, frameStretched)
    if cv2.waitKey(1) & 0xFF == ord('q') : 
        break

#cam.release()
#out.release()
cv2.destroyAllWindows()