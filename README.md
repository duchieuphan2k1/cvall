![alt text](static/CVall_logo.png)

## 1. Installation
- Python Version: 3.8

- Cuda Version: 12.0

Download ```base_model.zip``` via [Google Drive](https://drive.google.com/file/d/16e59TO1yJAW1vrr9gFwl1gjzNB4WUod_/view?usp=sharing). Then unzip it and put into ./models folder

### For Windows, run these command on command line:

```set PYTHONPATH=%PYTHONPATH%;damo_yolo2```

```set PYTHONPATH=%PYTHONPATH%;FastSAM2```

```set PYTHONPATH=%PYTHONPATH%;.```

```pip install -r requirements.txt```

### For Ubuntu, run these command on terminal:

```export PYTHONPATH=$PYTHONPATH:damo_yolo2:FastSAM2:.``` 

```pip install -r requirements.txt```

## 2. Run App

### For both Windows and Ubuntu, run this command on cmd/terminal resprectively:

```python app.py```

Then you can access the web system via this address: http://127.0.0.1:8091

## 3. To be updated
