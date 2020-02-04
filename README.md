![Demo 2](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/demo/garb_demo_3.gif)

# Garbage detection using PyTorch and YoloV3

For more information, look at [this](https://medium.com/maarten-sukel/garbage-object-detection-using-pytorch-and-yolov3-d6c4e0424a10) medium post.

PyTorch implementation of a garbage detection model. This repository contains all code for predicting/detecting and evaulating the model.

This repository combines elements from:
* https://github.com/zhaoyanglijoey/yolov3
* https://github.com/ultralytics/yolov3

![Demo 1](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/demo/demo_1.png)

Test and prediction code for a garbage object detection

## Installation

To install all required libaries:
```
pip install -r requirements.txt
```

## Predictions

Several different weights and configs are available at: https://drive.google.com/open?id=1DjeNxdaF7AW3Nu54_3oRw_1SeYJtOvNL. Some also have the testing data.

### Pre trained weights

| Name | Classes          | Test data  |
| ------------- |:-------------:| -----:|
| 3 classes| cardboard, garbage_bags and containers| Yes |
| cigarettes | cigarette     |  Yes|
| 12 classes| container_small, garbage_bag, cardboard, matras, christmas_tree, graffiti, pole, face_privacy_filter and license_plate_privacy_filter, construction_toilet, construction_container, construction_shed  |   No|


### Run predictions
To run predictions, download the cfg and weights from https://drive.google.com/open?id=1DjeNxdaF7AW3Nu54_3oRw_1SeYJtOvNL and put them in the correct folders. 

Then for example run the following the make a prediction on a file using CPU:

```
python detector_garb.py -i samples/input5_frame11.jpg -o output
```

Or to realtime detect on your webcam using GPU: (CUDA must be installed)
```
python detector_garb.py -i 0 --webcam --video -o ./webcam_output/ --cuda
```

### Docker

To run code in docker
```
docker-compose build
docker-compose up
```

## Test

For testing download data from:
https://drive.google.com/drive/folders/1DjeNxdaF7AW3Nu54_3oRw_1SeYJtOvNL

The garbage bags, containers and cardboard dataset contains 804 images and label files. A smaller dataset with annotations of cigarettes is also available.

To run test execute the following code:

```
python test.py
```

| Class           | Images | Targets | P     | R     | mAP   | F1    |
|-----------------|--------|---------|-------|-------|-------|-------|
| all             | 115    | 579     | 0.242 | 0.941 | 0.875 | 0.376 |
| container_small | 115    | 180     | 0.38  | 0.989 | 0.979 | 0.549 |
| garbage_bag     | 115    | 223     | 0.212 | 0.964 | 0.875 | 0.348 |
| cardboard       | 115    | 176     | 0.122 | 0.869 | 0.77  | 0.231 |



![test_example](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/test_batch0.jpg)

The model with 12 classes has been trained on a larger collection. The test results are below.

|Class |Images|Targets|P |R |mAP |F1|
|---------------------|--------|---------|-------|-------|-------|-------|
|all|111|490|0.232|0.913|0.855|0.365|
|container_small|111|91|0.407|0.956|0.948|0.57|
|garbage_bag|111|82|0.192|0.78|0.725|0.308|
|cardboard|111|61|0.201|0.885|0.829|0.327|
|matras|111|3|0.273|1|1|0.429|
|kerstboom|111|11|0.147|0.909|0.848|0.253|
|graffiti|111|34|0.15|1|0.98|0.262|
|amsterdammertje|111|39|0.236|1|0.989|0.382|
|face_privacy_filter|111|63|0.155|0.825|0.64|0.261|
|license_plate_privacy_filter|111|79|0.226|0.797|0.615|0.352|
|construction_toilet|111|5|0.235|0.8|0.7|0.364|
|construction_container|111|16|0.246|1|0.982|0.395|
|construction_shed|111|6|0.316|1|1|0.48|


## Training
For training a new model look at:

https://github.com/maartensukel/yolov3-garbage-object-detection-training

This is the training loss of 1600 images with 12 classes:
![test_example](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/loss.png)
