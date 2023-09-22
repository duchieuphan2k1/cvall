import json
import os
import signal

class StatusHandler:
    def __init__(self, model_name):
        self.status_folder = "models_status"
        if not os.path.exists(self.status_folder):
            os.mkdir(self.status_folder)
        self.model_name = model_name
        self.status_file = os.path.join(self.status_folder, "{}.json".format(model_name))

        if os.path.exists(self.status_file):
            self.status_template = json.load(open(self.status_file, 'r'))
        else:
            self.status_template = {
                'pid_id': None,
                'training_status': None,
                'training_info': [{'epoch': 0, 'accuracy': 0}],
                'total_epochs': 1500
            }

    def update_info(self, total_epochs):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['total_epochs'] = total_epochs
        self.save_status()

    def create_status(self, pid_id):
        self.status_template['pid_id'] = pid_id
        self.status_template['training_status'] = 'training'
        self.save_status()

    def update_process(self, epoch, accuracy):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['training_info'].append({"epoch": epoch, "accuracy": accuracy})
        self.save_status()

    def terminate_process(self):
        self.status_template = json.load(open(self.status_file, 'r'))
        self.status_template['training_status'] = 'terminated'
        os.kill(self.status_template['pid_id'], signal.SIGTERM)
        self.save_status()
    
    # def complete_process(self):

    #     model_workdir = "workdir/{}/base".format(self.model_name)

    #     bestmap_models = []
    #     all_ckpt = os.listdir(model_workdir)
    #     for ckpt in all_ckpt:
    #         if 'best_map' in ckpt:
    #             bestmap_models.append(ckpt)
        
    #     bestmap_models.sort()
    #     to_saved = bestmap_models[-1]
    #     to_saved_path = os.path.join('models', "{}.pth".format(self.model_name))
    
    #     if os.path.exists(to_saved_path):
    #         shutil.rmtree(to_saved_path)
    #     shutil.copy(os.path.join(model_workdir, to_saved), to_saved_path)

    #     self.status_template = json.load(open(self.status_file, 'r'))
    #     self.status_template['training_status'] = 'completed'
    #     self.save_status()

    def save_status(self):
        # Serializing json
        json_object = json.dumps(self.status_template, indent=4)
        
        # Writing to sample.json
        with open(self.status_file, "w") as outfile:
            outfile.write(json_object)
