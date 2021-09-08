from flask import Flask
from flask import url_for, request, render_template, abort, Response
from werkzeug.utils import secure_filename
import base64
import cv2
from detection_model import detect, detect_video
import core.utils as utils
import numpy as np

client = Flask(__name__, static_folder="Client_static", template_folder="Client_templates")
client.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 3
client.config['UPLOAD_PATH'] = r'./Photo/Upload Folder'
client.config['DETECTION_PATH'] = r'./detections'

# this function return main page of website
@client.route("/main", methods=['GET', 'POST'])
@client.route("/", methods=['GET', 'POST'])
def main_page():
    return render_template("client.html")


@client.route("/toMain", methods=['GET'])
def to_main_page():
    return {"status": True, "redirect": url_for("main_page")}


@client.route('/uploadImage', methods=['POST'])
def upload_image():
    file = request.files['image']
    file_name = secure_filename(file.filename)
    upload_file = file.read()
    np_img = np.frombuffer(upload_file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    path = client.config['UPLOAD_PATH'] + "/" + file_name
    cv2.imwrite(path, img)
    detect.main(weights='./checkpoints/custom-416', images=img, image_name=file_name)

    return {"status": True, "redirect": url_for("result_image", filename=file_name)}


@client.route('/uploadVideo', methods=['POST'])
def video_prediction():
    try:
        file = request.files['video']
        file_name = secure_filename(file.filename)
        path = client.config['UPLOAD_PATH'] + "/" + file_name
        file.save(path)
    except:
        return {"status": True, "redirect": url_for("video_page", filename="camera")}
    try:
        return {"status": True, "redirect": url_for("video_page", filename=file_name)}
    except (FileNotFoundError, NameError):
        return abort(404)


@client.route('/uploadVideo/<filename>', methods=['POST', 'GET'])
def video_page(filename):
    return render_template("video_result.html", entries=[filename])


@client.route('/result/image/<filename>', methods=['POST', 'GET'])
def result_image(filename):
    file_name_new, extension = filename.split(".")

    list_NP = utils.image_to_text()
    path = "./detections/" + file_name_new + "_bb." + extension
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        encoded_string = encoded_string.decode("ascii")
        str_add = "data:image/" + extension + ";base64,"
        encoded_string = str_add + encoded_string
    utils.delete_detection_files()
    return render_template("photo_result.html", entries=[encoded_string, list_NP])


@client.route('/result/video/<filename>', methods=['POST', 'GET'])
def result_video(filename):
    if filename != "camera":
        path = client.config['UPLOAD_PATH'] + "/" + filename
        return Response(detect_video.main(weights='./checkpoints/custom-416', video=path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(detect_video.main(weights='./checkpoints/custom-416'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    client.config['TEMPLATES_AUTO_RELOAD'] = True
    client.run(host='0.0.0.0', debug=True)
