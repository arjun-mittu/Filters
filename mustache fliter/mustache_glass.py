import cv2
import dlib
import numpy as np
import os 
import glob
imgMustache = cv2.imread("msoustache.png",-1)
mask=imgMustache[:,:,3]
mask_inv=cv2.bitwise_not(mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

image = cv2.imread('rock.jpg')
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
width=abs(3*(landmarks[31]-landmarks[35]))
width_mustache=np.squeeze(np.asarray(width))
print(width_mustache[0])
height_mustache = int(width_mustache[0] * origMustacheHeight / origMustacheWidth) - 10
print(height_mustache)
#cv2.imshow('Result', image_with_landmarks)
mustache = cv2.resize(imgMustache, (width_mustache[0],height_mustache), interpolation = cv2.INTER_AREA)
mask=cv2.resize(mask, (width_mustache[0],height_mustache), interpolation = cv2.INTER_AREA)
mask_inv=cv2.resize(mask_inv, (width_mustache[0],height_mustache), interpolation = cv2.INTER_AREA)
#cv2.imshow('Result', mask_inv)
y_use=np.squeeze(np.asarray(landmarks[33]))
y1 = int(y_use[1] - (height_mustache/2)) + 10
y2 = int(y1 + height_mustache)
x_use=np.squeeze(np.asarray(landmarks[51]))
#print(x_use[0])
x_use_1=x_use[0]
print(x_use_1)
x1 =int( x_use_1-(width_mustache[0]/2))
print(x1)
x2 = int(x1 + width_mustache[0])
roi = image[y1:y2, x1:x2]
roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
image[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)
cv2.imshow("with filter",image)
cv2.waitKey(0)
cv2.destroyAllWindows()