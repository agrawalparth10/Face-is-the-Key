
import numpy as np 
import cv2 
import pickle 
import time


class Recognizer():
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        time.sleep(2.0)
        ProtoPath = "face_detection_model/deploy.prototxt"
        ModelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        EmbedderPath = "face_detection_model/openface_nn4.small2.v1.t7"

        self.detector = cv2.dnn.readNetFromCaffe(ProtoPath,ModelPath)
        self.embedder = cv2.dnn.readNetFromTorch(EmbedderPath)
        self.classifier = pickle.loads(open("./pickle/classifier.pickle","rb").read())
        self.labels = pickle.loads(open("./pickle/label.pickle","rb").read())

    def recognize(self):
        total = 0
        user = 0
        unknown = 0
        while(total < 20):
            ret, image = self.capture.read()
            (h, w) = image.shape[:2]


            image_blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

            self.detector.setInput(image_blob)
            detections = self.detector.forward()

            index = np.argmax(detections[0,0,:,2])
            box = detections[0,0,index,3:7] * np.array([w,h,w,h]) 
            x1,y1,x2,y2 = box.astype(int)
            
            face_input = image[y1:y2,x1:x2]

            (fH, fW) = face_input.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face_input, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(faceBlob)
            embedding = self.embedder.forward()

            preds = self.classifier.predict_proba(embedding)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = self.labels.classes_[j]

            text = "{}".format(name)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            face = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            face = cv2.putText(face, text, (x1, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            cv2.imshow("face_detection", face)
            if(name == "user"):
                user += 1
            else:
                unknown +=1
            total+=1

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        self.capture.release()
        cv2.destroyAllWindows
        if(user > unknown):
            return True
        else:
            return False
