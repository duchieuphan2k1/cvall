from damo_yolo2.damo.apis import Trainer
from damo_yolo2.damo.config.base import parse_config
from utils.handle_path import PathHandler
from evaluation.predict_dataset import PredictDataset
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type = str, help = 'model name')
    args = parser.parse_args()
    path_handler = PathHandler()
    config_file_path = path_handler.get_config_path_by_name(args.model_name)
    config = parse_config(config_file_path)
    trainer = Trainer(config, PredictDataset, None)
    print("==========================")
    trainer.train(local_rank=0)