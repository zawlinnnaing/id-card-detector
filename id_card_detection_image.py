
# Import packages
from utils import visualization_utils as vis_util
from utils import label_map_util
import traceback
import argparse
import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
import sys

from PIL import Image


def get_images(img_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    if os.path.isfile(img_path):
        for ext in exts:
            if img_path.endswith(ext):
                files.append(img_path)
    else:
        for parent, _, filenames in os.walk(img_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
    print('Found {} images'.format(len(files)))
    return files


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Detecting and cropping ID card from image")
    parser.add_argument(
        '--img', help="Image file or directory to be processed", action="store", default="test_images")

    parser.add_argument(
        "--output_dir", help="Output directory for cropped image", action="store", default="output")

    return parser.parse_args()
    # This is needed since the notebook is stored in the object_detection folder.


sys.path.append("..")

args = argument_parser()

# Import utilites

# Name of the directory containing the object detection module we're using
CWD_PATH = os.getcwd()

MODEL_NAME = 'model'
IMAGE_NAME = args.img
IMAGE_EXTS = ['gif', 'jpg', 'png', 'jpeg', 'tiff']
OUTPUT_DIR = os.path.join(CWD_PATH, args.output_dir)

# Grab path to current working directory

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Remove output dir
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

images = get_images(PATH_TO_IMAGE)


for image_fn in images:
    try:
        print(f"running images {image_fn}")
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(image_fn)

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # print("confidence scores", scores.shape, boxes.shape, classes.shape)
        # Draw the results of the detection (aka 'visulaize the results')
        image, array_coord, box_to_display_str_map = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.60)

        ymin, xmin, ymax, xmax = array_coord

        for key, value in box_to_display_str_map.items():
            print("box to display str map", key, value)
        shape = np.shape(image)
        im_width, im_height = shape[1], shape[0]
        (left, right, top, bottom) = (xmin * im_width,
                                      xmax * im_width, ymin * im_height, ymax * im_height)

        # Using Image to crop and save the extracted copied image
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_fn))
        print(f"output dir {output_path}")
        im = Image.open(image_fn)
        im.crop((left, top, right, bottom)).save(output_path, quality=95)
    except Exception as e:
        traceback.print_exc()
        print("exception found for image", image_fn)
        continue
