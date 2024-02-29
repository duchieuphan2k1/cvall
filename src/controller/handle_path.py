import os
from src.controller.handle_config import ConfigHandler

class PathHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.general_config = self.cfg_handler.get_general_config()
    
    def get_model_params_path_by_name(self, model_name):
        model_path = self.get_model_dir_by_name(model_name)
        model_yaml_name = self.general_config.path.model_yaml_name
        yaml_path = os.path.join(model_path, model_yaml_name)
        return yaml_path
    
    def get_model_dir_by_name(self, model_name):
        models_dir = self.general_config.path.models_dir
        model_path = os.path.join(models_dir, model_name)
        return model_path
    
    def get_model_status_path(self, model_name):
        model_path = self.get_model_dir_by_name(model_name)
        status_file_name = self.general_config.path.model_status_file_name
        status_file_path = os.path.join(model_path, status_file_name)
        return status_file_path
    
    def get_model_exp_dir_by_name(self, model_name):
        exp_dir = self.general_config.path.evaluation_dir
        model_exp_path = os.path.join(exp_dir, model_name)
        return model_exp_path
    
    def get_ckpt_dir_by_name(self, model_name):
        models_dir = self.general_config.path.models_dir
        model_ckpt_name = self.general_config.path.model_ckpt_name

        model_path = os.path.join(models_dir, model_name)
        ckpt_dir = os.path.join(model_path, model_ckpt_name)
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        return ckpt_dir 

    def get_ckpt_path_by_name(self, model_name):
        models_dir = self.general_config.path.models_dir
        model_ckpt_name = self.general_config.path.model_ckpt_name

        model_path = os.path.join(models_dir, model_name)
        ckpt_dir = os.path.join(model_path, model_ckpt_name)

        if not os.path.exists(ckpt_dir):
            return None

        for ckpt in os.listdir(ckpt_dir):
            if ".pth" in ckpt or ".onnx" in ckpt or ".trt" in ckpt:
                ckpt_name = ckpt
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        return ckpt_path

    def get_config_dir_by_name(self, model_name):
        models_dir = self.general_config.path.models_dir
        model_configs_name = self.general_config.path.model_configs_name

        model_path = os.path.join(models_dir, model_name)
        config_dir = os.path.join(model_path, model_configs_name)

        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        return config_dir
        
    def get_config_path_by_name(self, model_name):
        models_dir = self.general_config.path.models_dir
        model_configs_name = self.general_config.path.model_configs_name

        model_path = os.path.join(models_dir, model_name)
        config_dir = os.path.join(model_path, model_configs_name)

        if not os.path.exists(config_dir):
            return None
        
        for cfg in os.listdir(config_dir):
            if ".py" in cfg:
                cfg_name = cfg
        config_path = os.path.join(config_dir, cfg_name)
        return config_path
    
    def get_input_demo_path(self):
        path = os.path.join("static", self.general_config.path.demo_dir_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    
    def get_output_demo_path(self):
        path = os.path.join("static", self.general_config.path.demo_output_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_labelme_segmentation_path(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.labelme_segment_annotation_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_dataset_path_by_name(self, dataset_name):
        path = os.path.join(self.general_config.path.datasets_dir, dataset_name)
        return path
    
    def get_image_path_by_name(self, dataset_name):
        print("dataset_name", dataset_name)
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.image_dir_name)
        return path
    
    def get_dataset_info_path_by_name(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.dataset_info_file)
        return path
    
    def get_labelme_annotation_path(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.labelme_annotation_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    
    def get_evaluation_folder(self, model_name, dataset_name):
        evaluation_dir = self.general_config.path.evaluation_dir
        if not os.path.exists(evaluation_dir):
            os.mkdir(evaluation_dir)
        
        model_eval_dir = os.path.join(evaluation_dir, model_name)
        if not os.path.exists(model_eval_dir):
            os.mkdir(model_eval_dir)

        dataset_eval_dir = os.path.join(model_eval_dir, dataset_name)
        if not os.path.exists(dataset_eval_dir):
            os.mkdir(dataset_eval_dir)
        
        return dataset_eval_dir
    
    def get_pred_labelme_dir(self, model_name, dataset_name):
        dataset_eval_dir = self.get_evaluation_folder(model_name, dataset_name)
        path = os.path.join(dataset_eval_dir, self.general_config.path.pred_labelme_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_pred_vis_dir(self, model_name, dataset_name):
        dataset_eval_dir = self.get_evaluation_folder(model_name, dataset_name)
        path = os.path.join(dataset_eval_dir, self.general_config.path.pred_vis_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_result_file_path(self, model_name, dataset_name):
        dataset_eval_dir = self.get_evaluation_folder(model_name, dataset_name)
        path = os.path.join(dataset_eval_dir, self.general_config.path.all_results_file_name)
        return path
    
    def get_precision_plot_dir(self, model_name, dataset_name):
        dataset_eval_dir = self.get_evaluation_folder(model_name, dataset_name)
        path = os.path.join(dataset_eval_dir, self.general_config.path.precision_plot_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_object_dir(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.object_dir_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_coco_annotation_dir(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.coco_annotaton_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_coco_annotation_file(self, dataset_name):
        coco_annotation_dir = self.get_coco_annotation_dir(dataset_name)
        path = os.path.join(coco_annotation_dir, self.general_config.path.coco_annotaton_file)
        return path

    def get_background_set_by_name(self, bg_name):
        background_dir = self.general_config.path.background_dir
        path = os.path.join(background_dir, bg_name)
        return path
    
    def get_augment_config_path(self, augment_dataset_name):
        augment_dataset_dir = self.get_dataset_path_by_name(augment_dataset_name)
        path = os.path.join(augment_dataset_dir, self.general_config.path.augment_yaml_name)
        return path
    
    def get_info_plot_dir(self, dataset_name):
        dataset_path = self.get_dataset_path_by_name(dataset_name)
        path = os.path.join(dataset_path, self.general_config.path.info_plot_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_class_number_plot_path(self, dataset_name):
        info_plot_dir = self.get_info_plot_dir(dataset_name)
        path = os.path.join(info_plot_dir, self.general_config.path.class_number_plot_name)
        return path

    def get_size_plot_path(self, dataset_name):
        info_plot_dir = self.get_info_plot_dir(dataset_name)
        path = os.path.join(info_plot_dir, self.general_config.path.image_object_size_ratio)
        return path