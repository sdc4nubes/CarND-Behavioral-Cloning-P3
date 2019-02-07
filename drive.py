import argparse
import base64
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        # Convert to Numpy array
        image_array = np.asarray(image)
        # Extract the center of the image
        image_array = image_array[60:130,80:240,:]
        # Greyscale image
        image_array = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=.8, tileGridSize=(4,4))
        image_array = clahe.apply(np.uint8(image_array))
        # Apply histogram equalization
        image_array = cv2.equalizeHist(np.uint8(image_array))
        # Resize to 32x32 image
        image_array = cv2.resize(image_array, (32, 32))
        # Reshape image to 32x32x1 (model input size)
        image_array = image_array.reshape(32, 32, 1)
        # Predict Steering Angle from image
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        steering_angle *= abs(steering_angle)
        steering_angle = round(steering_angle, 2)
        # Get throttle value based on speed and steering angle
        throttle = get_throttle(throttle, steering_angle, speed)
        print("%5.2f" % steering_angle, "%4.2f" % throttle, "%5.2f" % float(speed))
        send_control(steering_angle, throttle)
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

def get_throttle(throttle, steer, speed):
    speed = float(speed)
    throttle = float(throttle)
    if speed <=10: return 1
    return max(1 - ((steer * 50) ** 2), .1)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)
    
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='model.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='run1',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
