import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def predict_image(saved_model_loaded, image_data, iou, score):
    infer = saved_model_loaded.signatures['serving_default']
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
        iou_threshold=iou,
        score_threshold=score
    )
    return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]


def main(weights, images, size=416, output="./detections/", iou=0.45, score=0.25, image_name=None):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = size

    # load model
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

    if not isinstance(images, list) and image_name != None:
        image_data = cv2.resize(images, (input_size, input_size))
        image_data = image_data / 255.

        images_data = np.reshape(image_data, (1, input_size, input_size, 3)).astype(np.float32)

        pred_bbox = predict_image(saved_model_loaded, images_data, iou, score)
        file_name, extension = image_name.split(".")
        image = utils.draw_bbox(images, pred_bbox, file_name=(file_name, extension))
        image.astype(np.uint8)
        path = output + file_name + '_bb.' + extension
        cv2.imwrite(path, image)

    else:
        for count, image_path in enumerate(images):
            original_image = cv2.imread(image_path)
            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.

            images_data = np.reshape(image_data, (1, input_size, input_size, 3)).astype(np.float32)

            pred_bbox = predict_image(saved_model_loaded, images_data, iou, score)
            file_name, extension = (image_path.split("/")[-1]).split(".")
            image = utils.draw_bbox(original_image, pred_bbox, file_name=(file_name, extension))
            image.astype(np.uint8)
            path = output + file_name + '_bb.' + extension
            cv2.imwrite(path, image)


# if __name__ == '__main__':
    # img_lst = []
    # for i in os.listdir("./Photo/Upload Folder"):
    #     try:
    #         if int(i.split(".")[0])<252 and int(i.split(".")[0])>199:
    #             img_lst.append("./Photo/Number Plate Detection(Kaggle)/"+i)
    #     except:
    #         pass
    # try:
    # img = cv2.imread("./Photo/Upload Folder/7.jpg")
    # main(weights='./checkpoints/custom-416', images=img, image_name="7.jpg")
        # path = r"./detections/Crop/license_plate_2060.png"
        # list_NP = utils.image_to_text(path)
        # print(list_NP)
        # img = cv2.imread(path)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
    # except SystemExit:
    #     pass
