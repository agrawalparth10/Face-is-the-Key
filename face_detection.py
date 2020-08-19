from imutils import paths
import numpy as np
import pickle
import imutils
import cv2 
import time
import os

ProtoPath = "face_detection_model/deploy.prototxt"
ModelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
EmbedderPath = "face_detection_model/openface_nn4.small2.v1.t7"
detector = cv2.dnn.readNetFromCaffe(ProtoPath,ModelPath) 
embedding = cv2.dnn.readNetFromTorch(EmbedderPath)
names = []
embeddings = []

imagePaths = list(paths.list_images("./dataset"))



for (_,imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(str(imagePath))

    (h, w) = image.shape[:2] 
    image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)


    detector.setInput(image_blob)
    detections = detector.forward()

    index = np.argmax(detections[0,0,:,2])
    box = detections[0,0,index,3:7] * np.array([w,h,w,h]) 
    x1,y1,x2,y2 = box.astype(int)
    face = image[y1:y2,x1:x2]

    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
        (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedding.setInput(faceBlob)
    embedding_val = embedding.forward()
    embeddings.append(embedding_val.flatten())
    names.append(name)


data = {"embeddings" : embeddings ,"names" : names}

f = open("./pickle/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close() 