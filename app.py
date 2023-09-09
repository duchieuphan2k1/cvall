from flask import (Flask, flash, Response, render_template, request, redirect, session, url_for, send_file)
from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
from damo_yolo.tools.demo import InferRunner
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'itjustademosodonothackitpls'

cfg_handle = ConfigHandler()
general_cfg = cfg_handle.get_general_config()

path_handler = PathHandler()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/check_dataset_name", methods=["POST"])
def check_dataset_name():
    dataset_name = request.form.get("dataset_name")
    all_datasets = os.listdir(general_cfg.path.datasets_dir)
    if dataset_name in all_datasets:
        return 'false'
    return 'true'

@app.route("/add_new_dataset", methods=["POST"])
def add_new_dataset():
    dataset_name = request.form.get("dataset_name")
    all_datasets = os.listdir(general_cfg.path.datasets_dir)
    if dataset_name in all_datasets:
        return 'false'
    return 'true'

@app.route("/train")
def train():
    return render_template("train.html")

@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

@app.route("/demo")
def demo():
    models_dir = general_cfg.path.models_dir
    models = os.listdir(models_dir)
    return render_template("demo.html", models=models, input_type=None, input_path=None)

@app.route("/infer_detection", methods = ['POST'])
def infer_detection():
    input_type = request.form.get("input_type")
    input_name = request.form.get("input_name")
    model_name = request.form.get("model_name")
    started_ckpt = path_handler.get_ckpt_path_by_name(model_name)
    started_config = path_handler.get_config_path_by_name(model_name)
    infer_runner = InferRunner(started_config, started_ckpt, path_handler.get_output_demo_path())

    demo_input_dir = path_handler.get_input_demo_path()
    input_path = os.path.join(demo_input_dir, input_name)

    if input_type == "image":
        inf_time = infer_runner.run_image(input_path)
    elif input_type == "video":
        inf_time = infer_runner.run_video(input_path)
    
    return str(round(inf_time, 4))

@app.route('/upload_file',methods = ['POST'])
def upload_file():
    if request.method == 'POST':  
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            if '.mp4' in filename or '.avi' in filename:
                input_type = 'video'
            else:
                input_type = 'image'

            filepath = os.path.join(path_handler.get_input_demo_path(), filename)
            file.save(filepath)

            models_dir = general_cfg.path.models_dir
            models = os.listdir(models_dir)
            return render_template("demo.html", models=models, input_type=input_type, input_name=filename)

@app.route('/download_file',methods = ['GET'])
def download_file():
    file_name = request.args.get("file_name")
    file_path = os.path.join(path_handler.get_output_demo_path(), file_name)
    return send_file(file_path, download_name=file_name)

app.run(port=8091, debug=True)