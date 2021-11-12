import tensorflow as tf
import numpy as np
from PIL import Image

from backend.config import id2name
from backend.animal_config import category_index

PATH_TO_CKPT = 'models/frozen_inference_graph_human.pb'

def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.compat.v1.Session(graph=detection_graph)
                return sess, detection_graph



def inference(sess, detection_graph, img_arr, conf_thresh=0.5):
    # with detection_graph.as_default():
    #     with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(img_arr, axis=0)
    (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

    height, width, _ = img_arr.shape
    results = []
    for idx, class_id in enumerate(classes[0]):
        conf = scores[0, idx]
        if conf > conf_thresh:
            bbox = boxes[0, idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": id2name[class_id],
                            "conf": str(conf),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })

    # Animal detection
    ANIMAL_MODEL_PATH = '/Users/ph/Documents/Study/SJSU/298B/smartfarm_app/models/animal_saved_model'
    model = tf.saved_model.load(ANIMAL_MODEL_PATH)

    # image_np = np.array(Image.open('/content/drive/MyDrive/Night_animals_detection/dataset/test/mule_1d60e51c371ab2b2.jpg'))
    # image = np.asarray(image_np)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(img_arr)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    for idx, class_id in enumerate(output_dict['detection_classes']):
        conf = output_dict['detection_scores'][idx]
        if conf > 0.5:
            bbox = output_dict['detection_boxes'][idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": category_index[class_id],
                            "conf": str(conf),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })

    return {"results":results}

    