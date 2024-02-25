from flask import (Flask, flash, Response, render_template, request, redirect, session, url_for, send_file)
from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
from utils.handle_dataset import DatasetHandler
from utils.handle_model import ModelHandler
from utils.handle_status import StatusHandler
from data_augment.augment_dataset import DatasetAugment
from damo_yolo2.tools.demo import InferRunner
from FastSAM2.fastsam_inference import FastFAM_Infer
from evaluation.predict_dataset import PredictDataset
from damo_yolo2.damo.apis import Trainer
from damo_yolo2.damo.config.base import parse_config
import matplotlib.pyplot as plt
import json
import os
import cv2
import shutil
import ast
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'itjustademosodonothackitpls'

cfg_handle = ConfigHandler()
general_cfg = cfg_handle.get_general_config()

path_handler = PathHandler()
dataset_handler = DatasetHandler()
model_handler = ModelHandler()
fastsam_infer = FastFAM_Infer()
dataset_augment = DatasetAugment()

@app.route("/")
def home():
    if not os.path.exists(general_cfg.path.datasets_dir):
        os.mkdir(general_cfg.path.datasets_dir)
    
    if not os.path.exists(general_cfg.path.models_dir):
        os.mkdir(general_cfg.path.models_dir)

    if not os.path.exists(general_cfg.path.background_dir):
        os.mkdir(general_cfg.path.background_dir)

    return render_template("home.html")

@app.route("/data/datasets")
def data():
    return render_template("data.html")

@app.route("/data/guide")
def data_guide():
    return render_template("data_guide.html")

@app.route("/review_dataset")
def review_dataset():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    return render_template("dataset_overview.html", dataset_name=dataset_name, dataset_info=dataset_info)

@app.route("/review_model")
def review_model():
    model_name = request.args.get("model_name")
    status_handler = StatusHandler(model_name)
    model_info = model_handler.get_model_info_by_name(model_name)
    model_status_info = status_handler.get_info()
    return render_template("model_overview.html", model_name=model_name, model_info=model_info, model_status_info=model_status_info)

@app.route("/train_model")
def train_model():
    model_name = request.args.get("model_name")
    status_handler = StatusHandler(model_name)
    model_info = model_handler.get_model_info_by_name(model_name)
    model_status_info = status_handler.get_info()
    return render_template("train_model.html", model_name=model_name, model_info=model_info, model_status_info=model_status_info)

@app.route("/get_training_status", methods=["POST"])
def get_training_status():
    model_name = request.form.get("model_name")
    status_handler = StatusHandler(model_name)
    model_status_info = status_handler.get_info()
    epochs = []
    results = []
    losses = []
    for epoch_result in model_status_info['training_info']:
        epochs.append(epoch_result['epoch'])
        results.append(epoch_result['valid_accuracy'])
        losses.append(epoch_result['train_loss'])
    
    if len(epochs) == 0:
        epochs = [0]
        results = [0]
        losses = [0]

    accuracy_image_path = os.path.join("static", general_cfg.path.accuracy_image_name)
    loss_image_path = os.path.join("static", general_cfg.path.loss_image_name)

    plt.plot(epochs, results, marker='o', label='Accuracy on test set')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(accuracy_image_path)
    plt.close()

    plt.plot(epochs, losses, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_image_path)
    plt.close()

    response_data = {
        "image_path": [accuracy_image_path, loss_image_path],
        "current_epoch": epochs[-1]
        }
    return response_data

@app.route("/start_train", methods=['POST'])
def start_train():
    model_name = request.form.get("model_name")
    trainset = request.form.get("trainset")
    testset = request.form.get("testset")
    number_epochs = int(request.form.get("number_epochs"))
    model_handler.update_num_epochs(model_name, number_epochs)
    dataset_handle = DatasetHandler()
    dataset_handle.labelme_to_coco(testset)
    dataset_handle.labelme_to_coco(trainset)

    using_params = {
        'train': {
            "img_dir": path_handler.get_image_path_by_name(trainset),
            "ann_file": path_handler.get_coco_annotation_file(trainset)
        },
        'test': {
            "img_dir": path_handler.get_image_path_by_name(testset),
            "ann_file": path_handler.get_coco_annotation_file(testset)
        }
    }
    ConfigHandler().dump_config_by_path(general_cfg.path.using_params_name, using_params)
    os.system("python wrap_training.py --model_name {}".format(model_name))
    return "Done Training"

@app.route("/terminate_training", methods=['POST'])
def terminate_training():
    model_name = request.form.get("model_name")
    status_handler = StatusHandler(model_name)
    status_handler.terminate_process()
    return ""

@app.route("/upload_data")
def upload_data():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    return render_template("upload_data.html", dataset_name=dataset_name, nbr_images=dataset_info['nbr_images'])

@app.route("/preview_data")
def preview_data():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    return render_template("dataset_image_preview.html", dataset_name=dataset_name, nbr_images=dataset_info['nbr_images'])

@app.route("/data_annotation")
def data_annotation():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    models_dir = general_cfg.path.models_dir
    models = os.listdir(models_dir)
    return render_template("data_annotation.html", dataset_name=dataset_name, nbr_images=dataset_info['nbr_images'], models=models, all_datasets=dataset_info['class_list'])

@app.route("/get_model_classes", methods=['POST'])
def get_model_classes():
    model_name = request.form.get('model_name')
    classes = model_handler.get_classes_by_name(model_name)
    return classes

@app.route("/auto_annotate", methods=['POST'])
def auto_annotate():
    dataset_name = request.form.get('dataset_name')
    model_name = request.form.get('model_name')
    selected_classes = request.form.get('selected_classes')
    selected_classes = ast.literal_eval(selected_classes)

    changed_names = request.form.get('changed_names')
    changed_names = ast.literal_eval(changed_names)

    print(selected_classes)
    predict_dataset = PredictDataset(dataset_name, model_name, generate_annotation=1)
    predict_dataset.runs(selected_classes, changed_names)
    return "Done"

@app.route("/get_auto_annotation_progress", methods=['POST'])
def get_auto_annotation_progress():
    dataset_name = request.form.get('dataset_name')
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    nbr_auto_annotated = dataset_info['nbr_auto_annotated']
    nbr_images = dataset_info['nbr_images']
    percent = int(nbr_auto_annotated/nbr_images*100)
    return [percent]

@app.route("/get_evaluation_progress", methods=['POST'])
def get_evaluation_progress():
    dataset_name = request.form.get('dataset_name')
    model_name = request.form.get('model_name')
    pred_labelme_dir = path_handler.get_pred_labelme_dir(model_name, dataset_name)
    image_dir = path_handler.get_image_path_by_name(dataset_name)
    nbr_preded = len(os.listdir(pred_labelme_dir))
    nbr_images = len(os.listdir(image_dir))

    percent = int(nbr_preded/nbr_images*100)
    return [percent]

@app.route("/view_results", methods=['GET'])
def view_results():
    dataset_name = request.args.get("dataset_name")
    model_name = request.args.get("model_name")
    static_results_dir = os.path.join("static", general_cfg.path.results_dir_name)
    if os.path.exists(static_results_dir):
        shutil.rmtree(static_results_dir)
    os.mkdir(static_results_dir)
    precision_plot_dir = path_handler.get_precision_plot_dir(model_name, dataset_name)
    image_list = os.listdir(precision_plot_dir)
    path_list = []
    average_precision = json.load(open(path_handler.get_result_file_path(model_name, dataset_name), 'r'))['avg_precision']
    for image_name in image_list:
        img = cv2.imread(os.path.join(precision_plot_dir, image_name))
        cv2.imwrite(os.path.join(static_results_dir, image_name), img)
        path_list.append(os.path.join(static_results_dir, image_name))

    return render_template("view_results.html", path_list=path_list, average_precision=average_precision, dataset_name=dataset_name, model_name=model_name)

@app.route("/get_all_evaluations", methods=["POST", "GET"])
def get_all_evaluations():
    experiment_dir = general_cfg.path.evaluation_dir
    all_models = os.listdir(experiment_dir)
    evaluation_info = []
    for model_name in all_models:
        model_exp_path = os.path.join(experiment_dir, model_name)
        exp_datasets = os.listdir(model_exp_path)
        for dts in exp_datasets:
            result_file_path = path_handler.get_result_file_path(model_name, dts)
            if os.path.exists(result_file_path):
                avg_precision = json.load(open(result_file_path, 'r'))['avg_precision']
                evaluation_info.append({
                    'model_name': model_name,
                    'dataset_name': dts,
                    'avg_precision': avg_precision
                })
    return evaluation_info


@app.route("/review_label", methods=["POST"])
def review_label():
    dataset_name = request.form.get('dataset_name')
    img_dir = path_handler.get_image_path_by_name(dataset_name)
    labelme_dir = path_handler.get_labelme_annotation_path(dataset_name)
    os.system("labelme {} -o {} --autosave --nodata".format(img_dir, labelme_dir))
    dataset_handler.update_preparation_progress(dataset_name, 3)
    return "Finished Review!"

@app.route("/review_segmentation", methods=["POST"])
def review_segmentation():
    dataset_name = request.form.get('dataset_name')
    img_dir = path_handler.get_image_path_by_name(dataset_name)
    segment_dir = path_handler.get_labelme_segmentation_path(dataset_name)
    os.system("labelme {} -o {} --autosave --nodata".format(img_dir, segment_dir))
    # dataset_handler.update_preparation_progress(dataset_name, 3)
    return "Finished Review!"

@app.route("/upload_images", methods=["POST"])
def upload_images():
    dataset_name = request.form.get("dataset_name")
    print(dataset_name)
    all_files = request.files.getlist('file')
    image_path = path_handler.get_image_path_by_name(dataset_name)
    for file in all_files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(image_path, filename))

            file_extension = filename.split('.')[-1]
            if file_extension!="jpg":
                img = cv2.imread(os.path.join(image_path, filename))
                cv2.imwrite(os.path.join(image_path, filename.replace(file_extension, 'jpg')), img)
                os.remove(os.path.join(image_path, filename))

    dataset_handler.update_preparation_progress(dataset_name, 2)
    return redirect('/upload_data?dataset_name={}'.format(dataset_name))

@app.route("/upload_video", methods=["POST"])
def upload_video():
    dataset_name = request.form.get("dataset_name")
    video_fps = int(request.form.get("video_fps"))
    file = request.files['video_file']

    dataset_path = path_handler.get_dataset_path_by_name(dataset_name)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(dataset_path, filename))
        dataset_handler.extract_images(dataset_name, filename, video_fps)

    dataset_handler.update_preparation_progress(dataset_name, 2)
    return "Video Uploaded"

@app.route("/upload_background", methods=["POST"])
def upload_background():
    background_upload_set = request.form.get("background_upload_set")
    files = request.files.getlist('background_file[]')
    print(files)

    background_path = path_handler.get_background_set_by_name(background_upload_set)

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(background_path, filename))

            file_extension = filename.split('.')[-1]
            if file_extension!="jpg":
                img = cv2.imread(os.path.join(background_path, filename))
                cv2.imwrite(os.path.join(background_path, filename.replace(file_extension, 'jpg')), img)
                os.remove(os.path.join(background_path, filename))

    return "Background Images Uploaded"

@app.route("/data_explore", methods=['GET'])
def data_explore():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    number_plot_path, size_plot_path = dataset_handler.plot_info(dataset_name)

    static_number_plot_path = os.path.join("static", path_handler.general_config.path.class_number_plot_name)
    static_size_plot_path = os.path.join("static", path_handler.general_config.path.image_object_size_ratio)

    if os.path.exists(static_number_plot_path):
        os.remove(static_number_plot_path)
    if os.path.exists(static_size_plot_path):
        os.remove(static_size_plot_path)

    shutil.copyfile(number_plot_path, static_number_plot_path)
    shutil.copyfile(size_plot_path, static_size_plot_path)

    return render_template("explore_data.html", dataset_name=dataset_name, dataset_info=dataset_info)

@app.route("/data_augment", methods=['GET'])
def data_augment():
    dataset_name = request.args.get("dataset_name")
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    nbr_auto_annotated = dataset_info['nbr_auto_annotated']
    nbr_images = dataset_info['nbr_images']
    return render_template("augment.html", dataset_name=dataset_name, nbr_auto_annotated=nbr_auto_annotated, nbr_images=nbr_images)

@app.route("/start_augment", methods=['POST'])
def start_augment():
    # new_augment_config = {
    #     'color': {
    #         'MultiplyHueAndSaturation': [float(request.form.get('MultiplyHueAndSaturationMin')), float(request.form.get('MultiplyHueAndSaturationMax'))],
    #         'ChangeColorTemperature': [int(request.form.get('ChangeColorTemperatureMin')), int(request.form.get('ChangeColorTemperatureMax'))],
    #         'ChannelShuffle': float(request.form.get('ChannelShuffleMin')),
    #         'Add': [int(request.form.get('AddMin')), int(request.form.get('AddMax'))],
    #         'GammaContrast': [float(request.form.get('GammaContrastMin')), float(request.form.get('GammaContrastMax'))],
    #         'MultiplyBrightness': [float(request.form.get('MultiplyBrightnessMin')), float(request.form.get('MultiplyBrightnessMax'))]
    #         },
    #     'resize': {
    #         'small_scale': [int(request.form.get('small_scale_min')), int(request.form.get('small_scale_max'))],
    #         'medium_scale': [int(request.form.get('medium_scale_min')), int(request.form.get('medium_scale_max'))],
    #         'big_scale': [int(request.form.get('big_scale_min')), int(request.form.get('big_scale_max'))],
    #         'small_percent': int(request.form.get('small_percent')),
    #         'medium_percent': int(request.form.get('medium_percent'))
    #     },
    #     'geometry': {
    #         'PerspectiveTransform': [float(request.form.get('PerspectiveTransformMin')), float(request.form.get('PerspectiveTransformMax'))],
    #         'Rotate': [int(request.form.get('RotateMin')), int(request.form.get('RotateMax'))],
    #         'ShearX': [int(request.form.get('ShearXMin')), int(request.form.get('ShearXMax'))],
    #         'ShearY': [int(request.form.get('ShearYMin')), int(request.form.get('ShearYMax'))],
    #         'PiecewiseAffine': [int(request.form.get('PiecewiseAffineMin')), int(request.form.get('PiecewiseAffineMax'))]
    #     },
    #     'other': {
    #         'use_color': int(request.form.get('use_color')),
    #         'use_geometry': int(request.form.get('use_geometry')),
    #         'use_resize': int(request.form.get('use_resize')),
    #         'number_augment': int(request.form.get('number_augment')),
    #         'nbr_objects_per_image': [int(request.form.get('nbr_objects_per_image_min')), int(request.form.get('nbr_objects_per_image_max'))],
    #         'accepted_threshold': int(request.form.get('accepted_threshold')),
    #         'grayscale_ratio': float(request.form.get('grayscale_ratio'))
    #     }
    # }
    augment_dataset_name = request.form.get('augment_dataset_name')
    origin_dataset_name = request.form.get('origin_dataset_name')
    background_set_name = request.form.get('background_set_name')

    dataset_augment.create_augment_dataset(origin_dataset_name, augment_dataset_name)
    # dataset_augment.change_augment_config(augment_dataset_name, new_augment_config)
    dataset_augment.augment_dataset(augment_dataset_name, origin_dataset_name, background_set_name)
    dataset_handler.update_preparation_progress(origin_dataset_name, 4)
    return "Successfully Augment Data"

@app.route("/segment_data", methods=['POST'])
def segment_data():
    dataset_name = request.form.get("dataset_name")
    fastsam_infer.run_dataset(dataset_name)
    return "Completed Segmentation"

@app.route("/get_segmentation_progress", methods=['POST'])
def get_segmentation_progress():
    dataset_name = request.form.get('dataset_name')
    dataset_info = dataset_handler.get_info_by_name(dataset_name)
    nbr_auto_annotated = dataset_info['nbr_segmented']
    nbr_images = dataset_info['nbr_images']
    percent = int(nbr_auto_annotated/nbr_images*100)
    return [percent]

@app.route("/start_extract_objects", methods=['POST'])
def start_extract_objects():
    dataset_name = request.form.get("dataset_name")
    fastsam_infer.extract_objects(dataset_name)
    return "Completed Extracting Objects"

@app.route("/get_extract_progress", methods=['POST'])
def get_extract_progress():
    dataset_name = request.form.get('dataset_name')
    objects_info = dataset_handler.get_nbr_objects(dataset_name)
    nbr_objects = objects_info['nbr_objects']
    nbr_extracted_objects = objects_info['nbr_extracted_objects']
    percent = int(nbr_extracted_objects/nbr_objects*100)
    return [percent]

@app.route("/get_augment_progress", methods=['POST'])
def get_augment_progress():
    augment_dataset_name = request.form.get('augment_dataset_name')
    augment_info = dataset_augment.load_augment_config_by_name(augment_dataset_name)
    total_images = augment_info.other.number_augment
    current_nbr = len(os.listdir(path_handler.get_image_path_by_name(augment_dataset_name)))
    percent = int(current_nbr/total_images*100)
    return [percent]

@app.route("/check_dataset_name", methods=["POST"])
def check_dataset_name():
    dataset_name = request.form.get("dataset_name")
    all_datasets = os.listdir(general_cfg.path.datasets_dir)
    if dataset_name in all_datasets:
        return 'false'
    return 'true'

@app.route("/check_model_name", methods=["POST"])
def check_model_name():
    model_name = request.form.get("model_name")
    all_models = os.listdir(general_cfg.path.models_dir)
    if model_name in all_models:
        return 'false'
    return 'true'

@app.route("/check_background_name", methods=["POST"])
def check_background_name():
    background_set_name = request.form.get("background_set_name")
    all_background_sets = os.listdir(general_cfg.path.background_dir)
    if background_set_name in all_background_sets:
        return 'false'
    return 'true'

@app.route("/add_background_set", methods=["POST"])
def add_background_set():
    background_set_name = request.form.get("background_set_name")
    set_path = path_handler.get_background_set_by_name(background_set_name)
    os.mkdir(set_path)
    return 'true'

@app.route("/get_all_background_sets", methods=["POST"])
def get_all_background_sets():
    all_background_sets = os.listdir(general_cfg.path.background_dir)
    return all_background_sets

@app.route("/get_number_background_images", methods=["POST"])
def get_number_background_images():
    background_set = request.form.get("background_set")
    set_path = path_handler.get_background_set_by_name(background_set)
    nbr_images = len(os.listdir(set_path))
    return [nbr_images]

@app.route("/add_new_dataset", methods=["POST"])
def add_new_dataset():
    if request.method == 'POST':
        class_list = []
        for key in request.form:
            if 'class' in key:
                class_list.append(request.form.get(key))
        if len(class_list) == 0:
            class_list.append("any")

        dataset_secarino = request.form.get("dataset_secarino")
        dataset_name = request.form.get("dataset_name")
        dataset_type = request.form.get("dataset_type")
        dataset_decs = request.form.get("dataset_decs")
        dataset_handler.create_dataset(dataset_name, dataset_secarino, dataset_type, dataset_decs, class_list)
    
    return redirect('/data/datasets')

@app.route("/add_new_model", methods=["POST"])
def add_new_model():
    if request.method == 'POST':
        model_name = request.form.get("model_name")
        trainset = request.form.get("trainset")
        testset = request.form.get("testset")
        model_decs = request.form.get("model_decs")
        model_handler.create_model(model_name, trainset, testset, model_decs)
    
    return redirect('/train')

@app.route('/delete_dataset', methods=["POST"])
def delete_dataset():
    if request.method == 'POST':
        dataset_name = request.form.get("dl_dataset")
        dataset_handler.delete_dataset(dataset_name)
    return redirect("/data/datasets")

@app.route('/delete_model', methods=["POST"])
def delete_model():
    model_name = request.form.get("dl_model")
    model_handler.delete_model(model_name)
    return redirect("/train")

@app.route('/delete_evaluation', methods=["POST"])
def delete_evaluation():
    model_name = request.form.get("dl_model_name")
    dataset_name = request.form.get("dl_dataset_name")
    path = path_handler.get_evaluation_folder(model_name, dataset_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    return redirect("/evaluation")

@app.route("/get_all_datasets", methods=["POST", "GET"])
def get_all_datasets():
    return dataset_handler.get_all_info()

@app.route("/get_all_models", methods=["POST", "GET"])
def get_all_models():
    return model_handler.get_all_general_info()

@app.route("/train")
def train():
    all_trainset = dataset_handler.get_all_trainset()
    all_testset = dataset_handler.get_all_testset()
    return render_template("train.html", all_trainset=all_trainset, all_testset=all_testset)

@app.route("/evaluation")
def evaluation():
    all_testsets = dataset_handler.get_all_testset()
    models_dir = general_cfg.path.models_dir
    all_models = os.listdir(models_dir)    
    return render_template("evaluation.html", all_testsets=all_testsets, all_models=all_models)

@app.route("/start_evaluate", methods=["POST"])
def start_evaluate():
    dataset_name = request.form.get('dataset_name')
    model_name = request.form.get('model_name')
    predict_dataset = PredictDataset(dataset_name, model_name)
    predict_dataset.run_pred()
    dataset_handler.labelme_to_coco(dataset_name)
    predict_dataset.eval_results(plot=True)
    return "Done Evaluation"


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
    segment = int(request.form.get("segment"))
    tracking = int(request.form.get("tracking"))

    started_ckpt = path_handler.get_ckpt_path_by_name(model_name)
    started_config = path_handler.get_config_path_by_name(model_name)
    infer_runner = InferRunner(started_config, started_ckpt, path_handler.get_output_demo_path())

    demo_input_dir = path_handler.get_input_demo_path()
    input_path = os.path.join(demo_input_dir, input_name)

    if input_type == "image":
        inf_time = infer_runner.run_image(input_path, segment=segment, tracking=tracking)
    elif input_type == "video":
        inf_time = infer_runner.run_video(input_path, segment=segment, tracking=tracking)
    
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