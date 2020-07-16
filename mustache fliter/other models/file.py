import cv2
import numpy as np
import glob
import os
import dlib
imgMustache = cv2.imread("msoustache.png", -1)
orig_mask = imgMustache[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
# 68 point detector on face
predictor_path = "shape_predictor_68_face_landmarks.dat"
# face detection modal
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
cnn_face_detector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
frame=cv2.imread("rock.jpg")
dets = cnn_face_detector(frame, 1) 
shape = predictor(frame, d.rect)
mustacheWidth = abs(3 * (shape.part(31).x - shape.part(35).x))
mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
y1 = int(shape.part(33).y - (mustacheHeight/2)) + 10
y2 = int(y1 + mustacheHeight)
x1 = int(shape.part(51).x - (mustacheWidth/2))
x2 = int(x1 + mustacheWidth)
roi = frame[y1:y2, x1:x2]
roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)
cv2.imshow(frame)
cv2.waitKey(0)
cv2.destroyAllWindows()