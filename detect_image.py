
#####################-------------------------------------------- IMPORTING THE REQUIREMENTS --------------------------------------------------########################

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
import os


#####################---------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------########################

def detect_image():
  
    input_size = 416                          # input size

    image_path = './data/images/dog.jpg'      # path to image file
    WEIGHT_PATH = "./checkpoints/1"  # path to yolov4 weights
    SAVE_PATH = "./detections/detected_image/image.png"                # path to save detections

    iou_threshold=0.45                        # iou threshold for non-max suppression
    score_threshold=0.25                      # score threshold for non-max suppression
    
    saved_model_loaded = tf.saved_model.load(WEIGHT_PATH, tags=[tag_constants.SERVING])    # loading yolov4 weights

    detect = str(input("Detect all ?  type 'yes/no' "))
    if detect == "yes":
        print("Detecting all")
    elif detect == 'no':
        clas = str(input("Enter the class : "))
        print("Detecting {}".format(clas))
    else:
        print("Not a valid input")

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']                               # loading model signatures
    batch_data = tf.constant(images_data)
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
    )                                                                                      # performing non-max suppression
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    if detect == 'no':
        image = utils.draw_bbox(original_image, pred_bbox, clas)                           # draw bounding boxes for custom class
    elif detect == 'yes':
        image = utils.draw_bbox_all(original_image, pred_bbox)                             # draw bounding boxes for all classes
        
    image = Image.fromarray(image.astype(np.uint8))

    image.show()

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(SAVE_PATH, image)
        

if __name__ == '__main__':
    detect_image()
