
#####################-------------------------------------------- IMPORTING THE REQUIREMENTS --------------------------------------------------########################

import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np


#####################---------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------########################

def detect_video():

    input_size = 416                          # input size

    WEIGHT_PATH = "./checkpoints/1"  # path to yolov4 weights
    SAVE_PATH = "./detections/detected_video/results.avi"    # path to video file
    output_format = 'XVID'                    # output format

    iou_threshold=0.45                        # iou threshold for non-max suppression
    score_threshold=0.25                      # score threshold for non-max suppression

    saved_model_loaded = tf.saved_model.load(WEIGHT_PATH, tags=[tag_constants.SERVING])   # loading the weights
    infer = saved_model_loaded.signatures['serving_default']                              # loading model signatures

    type_detection = str(input("Video/Webcam ? type video/web : "))

    if type_detection == 'video':
        video_path = './data/video/nyc_times.mp4'
    elif type_detection == 'web':
        video_path = 0

    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if SAVE_PATH:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(SAVE_PATH, codec, fps, (width, height))

    detect = str(input("Detect all ?  type 'yes/no' "))
    if detect == "yes":
        print("Detecting all")
    elif detect == 'no':
        clas = str(input("Enter the class name that you want to detect : "))
        print("Detecting {}".format(clas))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Detection Completed !! Results are in the "detection" folder.')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()


        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(                          
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )                                                                                         # performing non-max supression
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        if detect == 'no':
            image = utils.draw_bbox(frame, pred_bbox, clas)                                       # draw bounding boxes for custom classes
        elif detect == 'yes':
            image = utils.draw_bbox_all(frame, pred_bbox)                                         # draw bounding boxes for all classes

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Detections", result)
        
        if SAVE_PATH:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_video()
