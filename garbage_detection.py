import os
import argparse 
from model.darknet import Darknet
import torch
import logging
from model.util import process_result, load_images, resize_image, cv_image2tensor, transform_result, create_batches,create_output_json, load_data_frame
import math
import pickle as pkl
import os.path as osp
from datetime import datetime
from torch.autograd import Variable

class GarbageImageClassifier:
    
    """
    
    Classification models

    Image to json output with detected objects
    
    """

    def __init__(self,cuda,obj_thresh = 0.5, nms_thresh = 0.4):
        
        curScriptPath = os.path.dirname(os.path.abspath(__file__)) # needed to keep track of the current location of current script ( although it is included somewhere else )

        self.cuda = cuda
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        if cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        print('Loading network...')
        self.model = Darknet(curScriptPath + "/cfg/yolov3_garb_test.cfg")
        self.model.load_weights(curScriptPath + '/weights/garb.weights')

        if self.cuda:
            self.model.cuda()

        self.model.eval()
        print('Network loaded')

        self.createLogger()
        self.logger.info("GarbageImageClassifier: Init")
        
    # ----
    
    def createLogger(self):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO) # SETTING: log level
        
        # logger handlers
        handler = logging.StreamHandler()
        # handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-4s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # ----

    def detect_image(self,path,colors=[(39, 129, 113), (164, 80, 133), (83, 122, 114)],classes=['container_small', 'garbage_bag', 'cardboard']):

        print('Loading input image(s)...')
        input_size = [int(self.model.net_info['height']), int(self.model.net_info['width'])]
        batch_size = int(self.model.net_info['batch'])

        imlist, imgs = load_images(path)
        print('Input image(s) loaded')

        img_batches = create_batches(imgs, batch_size)


        
        print('Detecting...')

        all_images_attributes = []

        for batchi, img_batch in enumerate(img_batches):
            start_time = datetime.now()
            img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
            img_tensors = torch.stack(img_tensors)
            img_tensors = Variable(img_tensors)
            if self.cuda:
                img_tensors = img_tensors.cuda()
            detections = self.model(img_tensors, self.cuda).cpu()
            detections = process_result(detections, self.obj_thresh, self.nms_thresh)
            if len(detections) == 0:
                continue

            detections = transform_result(detections, img_batch, input_size)

            boxes = []
            for detection in detections:
                boxes.append(create_output_json(img_batch, detection, colors, classes))

            images_attributes = {}
            images_attributes['frameMeta'] = {'width':input_size[1],'height':input_size[0]}
            images_attributes['detectedObjects'] = boxes

            images_attributes['counts'] = {x:0 for x in classes}
            images_attributes['counts']['total'] = 0
            
            for box in boxes:
                images_attributes['counts'][box['detectedObjectType']] +=1
                images_attributes['counts']['total'] +=1
            end_time = datetime.now()
            print('Detection finished in %s' % (end_time - start_time))
            images_attributes['mlDoneAt'] = str(end_time)
            images_attributes['mlTimeTaken'] = end_time - start_time

            all_images_attributes.append(images_attributes)

        return all_images_attributes

    # ----

    def detect_image_data_frame(self,data_frame,colors=[(39, 129, 113), (164, 80, 133), (83, 122, 114)],classes=['container_small', 'garbage_bag', 'cardboard']):

        print('Loading input image(s)...')
        input_size = [int(self.model.net_info['height']), int(self.model.net_info['width'])]
        batch_size = int(self.model.net_info['batch'])

        imgs = [load_data_frame(data_frame)]
        print('Input image(s) loaded')

        img_batches = create_batches(imgs, batch_size)

        print('Detecting...')

        all_images_attributes = []

        for batchi, img_batch in enumerate(img_batches):
            start_time = datetime.now()
            img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
            img_tensors = torch.stack(img_tensors)
            img_tensors = Variable(img_tensors)
            if self.cuda:
                img_tensors = img_tensors.cuda()
            detections = self.model(img_tensors, self.cuda).cpu()
            detections = process_result(detections, self.obj_thresh, self.nms_thresh)
            if len(detections) == 0:
                continue

            detections = transform_result(detections, img_batch, input_size)

            boxes = []
            for detection in detections:
                boxes.append(create_output_json(img_batch, detection, colors, classes))

            images_attributes = {}
            images_attributes['frameMeta'] = {'width':input_size[1],'height':input_size[0]}
            images_attributes['detectedObjects'] = boxes

            images_attributes['counts'] = {x:0 for x in classes}
            images_attributes['counts']['total'] = 0
            
            for box in boxes:
                images_attributes['counts'][box['detectedObjectType']] +=1
                images_attributes['counts']['total'] +=1
            end_time = datetime.now()
            print('Detection finished in %s' % (end_time - start_time))
            images_attributes['mlDoneAt'] = str(end_time)
            images_attributes['mlTimeTaken'] = end_time - start_time

            all_images_attributes.append(images_attributes)

        


        return all_images_attributes
