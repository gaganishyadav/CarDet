from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt
import cv2
import numpy as np

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array

from werkzeug.wrappers import Request, Response

app=Flask(__name__)
model=keras.models.load_model('Model.h5', compile=False)

model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics='accuracy')

@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def pred():
    img=request.files['imagefile']
    img_path="images/"+img.filename
    img.save(img_path)
    car=[]
    image=cv2.imread(img_path)
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    for i in rects:
        x, y, w, h = i
        bb={'x1':x,
             'y1':y,
             'x2':x+w,
             'y2':y+h
            }
    img_data=image[bb['y1']:bb['y2'],bb['x1']:bb['x2']]
    img_data=cv2.resize(img_data,(224,224))
    Guess=model.predict(img_data.reshape(1,224,224,3))
    if Guess[0]>0.5:
      car.append([bb,Guess[0]])
    else:
      pass
    test_img=cv2.imread(img_path)
    x=np.argmax(np.array(car)[:,1])
    pt1=(car[x][0]['x1'],car[x][0]['y1'])
    pt2=(car[x][0]['x2'],car[x][0]['y2'])
    plt.figure()
    plt.imshow(test_img)
    cv2.rectangle(test_img,pt1,pt2,(255, 0, 0),2)
    plt.figure()
    classification=car[np.argmax(np.array(car)[:,1])][1][0]*100
    plt.imshow(test_img);
    detectedpath="images/images.png"
    detectedpathjpg="images/images.jpg"
    plt.savefig(detectedpath)
    return render_template('index.html', prediction=classification, detectedimage=detectedpath)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
