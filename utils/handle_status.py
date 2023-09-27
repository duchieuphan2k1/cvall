import json
import os
import signal
from utils.handle_path import PathHandler

class StatusHandler:
    def __init__(self, model_name):
        self.path_handler = PathHandler()
        self.status_file = self.path_handler.get_model_status_path(model_name)
        self.sample = {
                'terminating': False,
                'training_status': None,
                'training_info': [],
            }
        if os.path.exists(self.status_file):
            self.status_template = json.load(open(self.status_file, 'r'))
        else:
            self.status_template = self.sample

    def get_info(self):
        if os.path.exists(self.status_file):
            return json.load(open(self.status_file, 'r'))
        else:
            return self.sample
    def create_status(self):
        self.status_template = self.sample
        self.status_template['training_status'] = 'training'
        self.save_status()

    def update_process(self, epoch, accuracy, train_loss):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['training_info'].append({"epoch": epoch, "valid_accuracy": accuracy, 'train_loss': train_loss})
        self.save_status()

    def terminate_process(self):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['terminating'] = True
        self.status_template['training_status'] = 'terminated'
        self.save_status()

    def complete_training(self):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['training_status'] = 'completed'
        self.save_status()

    def save_status(self):
        json_object = json.dumps(self.status_template, indent=4)
        with open(self.status_file, "w") as outfile:
            outfile.write(json_object)
