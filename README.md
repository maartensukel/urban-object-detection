# Garbage detection using pytorch and YoloV3
## (Placeholder, work in progress)

[![Demo](https://img.youtube.com/vi/eP9xmQHbYCM/0.jpg)](https://www.youtube.com/watch?v=eP9xmQHbYCM)

Test and prediction code for a garbage object detection

## Predictions
To run predictions, download the cfg and weights from https://drive.google.com/open?id=1X62NUWxKD_hu9Z0ORFf25yLFkDqDAR0F

Then for example run the following the make a prediction:

```
python detector_garb.py -i samples/input5_frame281.jpg -o output --cuda
```

## Test

To be added



For training a new model look at:

https://github.com/maartensukel/yolov3-garbage-object-detection-training

## TODO:

Add test, with test results to readme. Add (test) data. Make readme more clear. Clean code. Clean data from personal information. Make public.