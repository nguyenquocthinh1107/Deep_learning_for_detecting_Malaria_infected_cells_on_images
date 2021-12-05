import tensorflow as tf
import cv2
import label_map_util 
import visualization_utils as viz_utils
import numpy as np
import pybase64
from yolov4 import yolov4_detect
import matplotlib.pyplot as plt

# some constant variables for RESNET
PATH_TO_LABELS = 'label_map.pbtxt'
MIN_CONF_THRESH = float(0.20)
PATH_TO_SAVED_MODEL = "resnet"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

# some constant variables for YOLOv4
# PATH_TO_CONFIG_FILE = 'yolov4/yolov4-custom.cfg'
# PATH_TO_DATA_FILE = 'yolov4/obj.data'
# PATH_TO_WEIGHTS_FILE = 'yolov4/yolov4-custom_last.weights'

def Decode(image):
    imgdata = pybase64.b64decode(image)
    image_out = np.asarray(bytearray(imgdata), dtype="uint8")
    image_out = cv2.imdecode(image_out, cv2.IMREAD_COLOR)
    return image_out

def detectWithResnet(image):
    print('Loading Resnet...')
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(image_with_detections,
                                                        detections['detection_boxes'],
                                                        detections['detection_classes'],
                                                        detections['detection_scores'],
                                                        category_index,
                                                        use_normalized_coordinates=True,
                                                        max_boxes_to_draw=200,
                                                        min_score_thresh=.30,
                                                        agnostic_mode=False)
    
    cv2.imwrite('static/output.png', image_with_detections)

def detectWithYOLOv4(image):
    print('Loading YOLOv4...')
    image_out, detections = yolov4_detect.image_detection(image, thresh=0.3)
    cv2.imwrite('static/output.png', image_out)

def detectWithUNET(image):
    print('Loading UNET...')
    # cv2.imwrite('static/input.png', image)
    # image_input = imageio.imread('static/input.png')

    # img_test = imageio.imread('/home/quocthinh/Study/SSD_MOBILENET_HELMET_DETECTION-main/Falciparum_24.png')
    model = tf.keras.models.load_model('unet/unet_singleclass_800x600.h5')
    net_input = np.expand_dims(image, axis=0)
    my_preds = model.predict(net_input, verbose=0)

    image_out = my_preds.reshape(512, 512)
    # image_out = cv2.convertScaleAbs(image_out, alpha=(255.0))

    plt.imsave('static/output.png', image_out)
    # cv2.imwrite('static/output.png', image_out)

