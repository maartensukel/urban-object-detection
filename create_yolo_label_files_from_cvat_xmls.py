# utf-8

# Script to convert .xml's that are outputted by cvat to yolov3 file format.

import os
from lxml import etree

def process_cvat_xml(xml_file, image_dir, output_dir):
    """
    Transforms a single XML in CVAT format to YOLO TXT files and download images when not in IMAGE_DIR

    :param xml_file: CVAT format XML
    :param image_dir: image directory of the dataset
    :param output_dir: directory of annotations with YOLO format
    :return:
    """
    KNOWN_TAGS = {'box', 'image', 'attribute'}

    if (image_dir is None):
        image_dir=os.path.join(output_dir,"data/obj")
        os.makedirs(image_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)
    cvat_xml = etree.parse(xml_file)
    basename = os.path.splitext( os.path.basename( xml_file ) )[0]
    current_labels =  {k: v for v, k in enumerate([line.rstrip('\n') for line in open('/home/maarten/Documents/projecten/pytorch/new_yolo/yolov3/data/garb.names')])}
    
    current_labels = {'container_small':0,'garbage_bag':1,'cardboard':2}

    traintxt = ""

    tracks= cvat_xml.findall( './/image' )
    


    for img_tag in cvat_xml.findall('image'):
        image_name = img_tag.get('name')
        width = int(img_tag.get('width'))
        height = int(img_tag.get('height'))
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            log.warn('{} image cannot be found. Is `{}` image directory correct?'.
                format(image_path, image_dir))

        unknown_tags = {x.tag for x in img_tag.iter()}.difference(KNOWN_TAGS)
        if unknown_tags:
            log.warn('Ignoring tags for image {}: {}'.format(image_path, unknown_tags))

        _yoloAnnotationContent = ""

        for box in img_tag.findall('box'):
            label = box.get('label')
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr'))

            if not label in current_labels:
                raise Exception('Unexpected label name {}'.format(label))

            labelid = current_labels[label]
            yolo_x = (xmin + ((xmax-xmin)/2))/width
            yolo_y = (ymin + ((ymax-ymin)/2))/height
            yolo_w = (xmax - xmin) / width
            yolo_h = (ymax - ymin) / height

            if len(_yoloAnnotationContent) != 0:
                    _yoloAnnotationContent += "\n"

            _yoloAnnotationContent += str(labelid)+" "+"{:.6f}".format(yolo_x) + " "+"{:.6f}".format(
                    yolo_y) + " "+"{:.6f}".format(yolo_w) + " "+"{:.6f}".format(yolo_h)

            
        anno_name = os.path.basename(os.path.splitext(image_name)[0] + '.txt')
        anno_path = os.path.join(output_dir+'/labels/', anno_name)
        
        if len(_yoloAnnotationContent) == 0:
                    _yoloAnnotationContent += "\n"
        
        print(anno_name,_yoloAnnotationContent)



        _yoloFile = open(anno_path, "w", newline="\n")
        _yoloFile.write(_yoloAnnotationContent)
        _yoloFile.close()


# directory with .xml for cvat
directory_xml_cvat = ''

# directory with images
directory_images = ''

# directory where output files will be stored
output_directory = ''

for xml in os.listdir(directory_xml_cvat):
        process_cvat_xml(directory_xml_cvat+xml, directory_images, output_directory)
