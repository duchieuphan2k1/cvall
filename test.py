from data_augment.augment_dataset import DatasetAugment

dag = DatasetAugment()
dag.augment_dataset("test_augment", "test01", "default")
