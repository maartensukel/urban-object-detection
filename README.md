## New version with yolov5: https://gitlab.com/odk/odk-frame-analyzer

![Demo 2](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/demo/garb_demo_3.gif)

# Urban object detection using PyTorch and YoloV3

For more information, look at [this](https://medium.com/maarten-sukel/garbage-object-detection-using-pytorch-and-yolov3-d6c4e0424a10) medium post.

PyTorch implementation of an urban object detection model. This repository contains all code for predicting/detecting and evaulating the model.

This repository combines elements from:
* https://github.com/zhaoyanglijoey/yolov3
* https://github.com/ultralytics/yolov3

![Demo 1](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/demo/demo_1.png)


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

| Class                        | Images | Targets | P     | R     | mAP   | F1    |
|------------------------------|--------|---------|-------|-------|-------|-------|
| all                          | 179    | 706     | 0.263 | 0.873 | 0.811 | 0.392 |
| container_small              | 179    | 142     | 0.521 | 0.972 | 0.97  | 0.678 |
| garbage_bag                  | 179    | 114     | 0.266 | 0.965 | 0.936 | 0.417 |
| cardboard                    | 179    | 78      | 0.132 | 0.962 | 0.89  | 0.232 |
| matras                       | 179    | 8       | 0.467 | 0.875 | 0.875 | 0.609 |
| kerstboom                    | 179    | 10      | 0.278 | 1     | 1     | 0.435 |
| graffiti                     | 179    | 73      | 0.185 | 0.932 | 0.885 | 0.308 |
| amsterdammertje              | 179    | 52      | 0.325 | 0.942 | 0.911 | 0.483 |
| face_privacy_filter          | 179    | 87      | 0.139 | 0.782 | 0.599 | 0.237 |
| license_plate_privacy_filter | 179    | 103     | 0.186 | 0.845 | 0.713 | 0.304 |
| construction_toilet          | 179    | 7       | 0.211 | 0.571 | 0.524 | 0.308 |
| construction_container       | 179    | 21      | 0.173 | 0.905 | 0.842 | 0.29  |

## Training
For training a new model look at:

https://github.com/maartensukel/yolov3-garbage-object-detection-training

This is the training loss of 1600 images with 12 classes:
![test_example](https://github.com/maartensukel/yolov3-pytorch-garbage-detection/raw/master/loss.png)
