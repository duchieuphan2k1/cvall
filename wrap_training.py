from damo_yolo2.damo.apis import Trainer
from damo_yolo2.damo.config.base import parse_config
from utils.handle_dataset import DatasetHandler

if __name__ == '__main__':
    config = parse_config('models/first_demo_model/configs/damoyolo_tinynasL25_S.py')
    dataset_handle = DatasetHandler()
    # dataset_handle.labelme_to_coco("augmented_test_01")
    # dataset_handle.labelme_to_coco("augmented_train_01")
    trainer = Trainer(config, None)
    print("==========================")
    trainer.train(local_rank=0)