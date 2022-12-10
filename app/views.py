from flask import Blueprint, render_template, request, send_file
import numpy as np
import cv2

from .model.final_model import Model

model = Model("app/model/")

views = Blueprint('views', __name__)

def process_file(file):
    img = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

@views.route('/')
def home():
    return render_template("base.html")

@views.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('image')
        img_np = process_file(f)
        domain = request.form.get('domain')

        prediction = model.predict_AB(img_np) if domain == "NIGHT2DAY" else model.predict_BA(img_np)
        prediction.save(f"app/temp_imgs/image.jpeg")
        return send_file("temp_imgs/image.jpeg", download_name=f.filename, as_attachment=True, mimetype="image/jpg")