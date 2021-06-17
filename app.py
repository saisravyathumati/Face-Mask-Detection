from flask import Flask,render_template,url_for,request,Response
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import winsound

app = Flask(__name__)

@app.route('/')
def home():
 return render_template('index.html')

def detect_and_predict_mask(frame, faceNet, maskNet):
 # grab the dimensions of the frame and then construct a blob
 # from it
 (h, w) = frame.shape[:2]
 blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
  (104.0, 177.0, 123.0))

 # pass the blob through the network and obtain the face detections
 faceNet.setInput(blob)
 detections = faceNet.forward()
 # initialize our list of faces, their corresponding locations,
 # and the list of predictions from our face mask network
 faces = []
 locs = []
 preds = []

 # loop over the detections
 for i in range(0, detections.shape[2]):
  # extract the confidence (i.e., probability) associated with
  # the detection
  confidence = detections[0, 0, i, 2]

  # filter out weak detections by ensuring the confidence is
  # greater than the minimum confidence
  if confidence > 0.5:
   # compute the (x, y)-coordinates of the bounding box for
   # the object
   box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
   (startX, startY, endX, endY) = box.astype("int")

   # ensure the bounding boxes fall within the dimensions of
   # the frame
   (startX, startY) = (max(0, startX), max(0, startY))
   (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

   # extract the face ROI, convert it from BGR to RGB channel
   # ordering, resize it to 224x224, and preprocess it
   face = frame[startY:endY, startX:endX]
   face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
   face = cv2.resize(face, (224, 224))
   face = img_to_array(face)
   face = preprocess_input(face)

   # add the face and bounding boxes to their respective
   # lists
   faces.append(face)
   locs.append((startX, startY, endX, endY))

 # only make a predictions if at least one face was detected
 if len(faces) > 0:
  # for faster inference we'll make batch predictions on *all*
  # faces at the same time rather than one-by-one predictions
  # in the above for loop
  faces = np.array(faces, dtype="float32")
  preds = maskNet.predict(faces, batch_size=32)

 # return a 2-tuple of the face locations and their corresponding
 # locations
 return (locs, preds)

@app.route('/predict')
def predict():
 prototxtPath = r"face_detector\\deploy.prototxt"
 weightsPath = r"face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
 faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
 maskNet = load_model("C:\\MiniProject3_2\\Face_Mask_Detection\\mask_detector.model")
 vs = cv2.VideoCapture(0)
 while True:
  pos,frame = vs.read()
  (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
  for (box, pred) in zip(locs, preds):
   (startX, startY, endX, endY) = box
   (mask, withoutMask) = pred
   label = "Mask" if mask > withoutMask else "No Mask"
   color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
   if(label=="No Mask"):
       winsound.Beep(400,1000)
   label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
   cv2.putText(frame, label, (startX, startY - 10),
     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
   cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
  frame = cv2.imencode('.JPEG', frame,[cv2.IMWRITE_JPEG_QUALITY,20])[1].tobytes()
  yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
 # vs.release()
 

@app.route('/video_feed')
def video_feed():
 return Response(predict(),mimetype='multipart/x-mixed-replace; boundary=frame')


if(__name__ =='__main__'):
 app.run(debug=True)
