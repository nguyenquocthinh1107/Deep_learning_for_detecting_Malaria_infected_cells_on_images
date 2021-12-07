import os
import glob
import random
import sys
sys.path.insert(1, '/home/quocthinh/Study/SSD_MOBILENET_HELMET_DETECTION-main/yolov4/darknet/')
import darknet
import time
import cv2
import numpy as np

PATH_TO_CONFIG_FILE = '/home/quocthinh/Study/SSD_MOBILENET_HELMET_DETECTION-main/yolov4/yolov4-custom.cfg'
PATH_TO_DATA_FILE = '/home/quocthinh/Study/SSD_MOBILENET_HELMET_DETECTION-main/yolov4/obj.data'
PATH_TO_WEIGHTS_FILE = '/home/quocthinh/Study/SSD_MOBILENET_HELMET_DETECTION-main/yolov4/yolov4-custom_best.weights'

def check_batch_shape(images, batch_size):
	"""
			Image sizes should be the same width and height
	"""
	shapes = [image.shape for image in images]
	if len(set(shapes)) > 1:
			raise ValueError("Images don't have same shape")
	if len(shapes) > batch_size:
			raise ValueError("Batch size higher than number of images")
	return shapes[0]


def load_images(images_path):
	"""
	If image path is given, return it directly
	For txt file, read it and return each line as image path
	In other case, it's a folder, return a list with names of each
	jpg, jpeg and png file
	"""
	input_path_extension = images_path.split('.')[-1]
	if input_path_extension in ['jpg', 'jpeg', 'png']:
			return [images_path]
	elif input_path_extension == "txt":
			with open(images_path, "r") as f:
					return f.read().splitlines()
	else:
			return glob.glob(
					os.path.join(images_path, "*.jpg")) + \
					glob.glob(os.path.join(images_path, "*.png")) + \
					glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
	width = darknet.network_width(network)
	height = darknet.network_height(network)

	darknet_images = []
	for image in images:
			image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image_resized = cv2.resize(image_rgb, (width, height),
																	interpolation=cv2.INTER_LINEAR)
			custom_image = image_resized.transpose(2, 0, 1)
			darknet_images.append(custom_image)

	batch_array = np.concatenate(darknet_images, axis=0)
	batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
	darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
	return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image, thresh):
	# Darknet doesn't accept numpy images.
	# Create one with image we reuse for each detect
	random.seed(3)  # deterministic bbox colors
	network, class_names, class_colors = darknet.load_network(
		PATH_TO_CONFIG_FILE,
		PATH_TO_DATA_FILE,
		PATH_TO_WEIGHTS_FILE
	)
	width = darknet.network_width(network)
	height = darknet.network_height(network)
	darknet_image = darknet.make_image(width, height, 3)

	# image = cv2.imread(image)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb, (width, height),
															interpolation=cv2.INTER_LINEAR)

	darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
	detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
	darknet.free_image(darknet_image)
	image = darknet.draw_boxes(detections, image_resized, class_colors)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
									thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
	image_height, image_width, _ = check_batch_shape(images, batch_size)
	darknet_images = prepare_batch(images, network)
	batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
																										image_height, thresh, hier_thresh, None, 0, 0)
	batch_predictions = []
	for idx in range(batch_size):
			num = batch_detections[idx].num
			detections = batch_detections[idx].dets
			if nms:
					darknet.do_nms_obj(detections, num, len(class_names), nms)
			predictions = darknet.remove_negatives(detections, class_names, num)
			images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
			batch_predictions.append(predictions)
	darknet.free_batch_detections(batch_detections, batch_size)
	return images, batch_predictions


def image_classification(image, network, class_names):
	width = darknet.network_width(network)
	height = darknet.network_height(network)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb, (width, height),
															interpolation=cv2.INTER_LINEAR)
	darknet_image = darknet.make_image(width, height, 3)
	darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
	detections = darknet.predict_image(network, darknet_image)
	predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
	darknet.free_image(darknet_image)
	return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
	"""
	YOLO format use relative coordinates for annotation
	"""
	x, y, w, h = bbox
	height, width, _ = image.shape
	return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
	"""
	Files saved with image_name.txt and relative coordinates
	"""
	file_name = os.path.splitext(name)[0] + ".txt"
	with open(file_name, "w") as f:
		for label, confidence, bbox in detections:
			x, y, w, h = convert2relative(image, bbox)
			label = class_names.index(label)
			f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))
