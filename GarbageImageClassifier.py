import os
import argparse
import logging
import sys
import torch


from darknet import Darknet


class GarbageImageClassifier:
    """
    Classification models

    Image to json output with detected objects
    """

    def __init__(self):

        curScriptPath = os.path.dirname(os.path.abspath(__file__)) # needed to keep track of the current location of current script ( although it is included somewhere else )

        parser = argparse.ArgumentParser(description='YOLOv3 object detection')
        parser.add_argument('-i', '--input', required=True, help='input image or directory or video')
        parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
        parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
        parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
        parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
        parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')
        parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
        parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

        self.args = parser.parse_args()

        if self.args.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        print('Loading network...')
        self.model = Darknet(curScriptPath + "/cfg/yolov3_garb_test.cfg")
        self.model.load_weights(curScriptPath + '/weights/garb.weights')

        if self.args.cuda:
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

    def detect_image(self,path):

        print('Loading input image(s)...')
        input_size = [int(model.net_info['height']), int(model.net_info['width'])]
        batch_size = int(model.net_info['batch'])

        imlist, imgs = load_images(args.input)
        print('Input image(s) loaded')

        img_batches = create_batches(imgs, batch_size)

        # load colors and classes
        colors = pkl.load(open("pallete", "rb"))
        classes = load_classes("cfg/garb.names")

        if not osp.exists(args.outdir):
            os.makedirs(args.outdir)

        start_time = datetime.now()
        print('Detecting...')

        for batchi, img_batch in tqdm(enumerate(img_batches)):
            img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
            img_tensors = torch.stack(img_tensors)
            img_tensors = Variable(img_tensors)
            if args.cuda:
                img_tensors = img_tensors.cuda()
            detections = model(img_tensors, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh)
            if len(detections) == 0:
                continue

            detections = transform_result(detections, img_batch, input_size)

            for detection in detections:
                draw_bbox(img_batch, detection, colors, classes,0,args.outdir)

            for i, img in enumerate(img_batch):
                save_path = osp.join(args.outdir, osp.basename(imlist[batchi*batch_size + i]))
                cv2.imwrite(save_path, img)
                print(save_path, 'saved')

        end_time = datetime.now()
        print('Detection finished in %s' % (end_time - start_time))
