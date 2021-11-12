# clone from https://github.com/AIZOOTech/flask-object-detection
from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import numpy as np
import base64
import io
import os

from backend.tf_inference import load_model, inference

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sess, detection_graph = load_model()

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html')

@app.route('/api/', methods=["POST"])
def main_interface():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

    image = base64.b64decode(base64_str)       
    img = Image.open(io.BytesIO(image))

    if(img.mode!='RGB'):
        img = img.convert("RGB")
    
    # convert to numpy array.
    img_arr = np.array(img)

    # do object detection in inference function.
    import time

    start = time.time()
    
    results = inference(sess, detection_graph, img_arr, conf_thresh=0.5)
    print(results)

    end = time.time()
    print(end - start)

    return jsonify(results)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
