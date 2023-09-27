from evaluation.predict_dataset import PredictDataset

predict_dataset = PredictDataset("test01", "base_model")
predict_dataset.run_pred()
predict_dataset.eval_results(plot=True)

