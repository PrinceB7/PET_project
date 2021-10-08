# evaluate the mask rcnn model on the PET dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN, load_image_gt, mold_image
from Mask_RCNN.mrcnn import model as modellib
from Mask_RCNN.mrcnn.utils import Dataset, compute_ap, compute_iou
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from tensorflow.keras.models import load_model


# class that defines and loads the PET dataset
class PETDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "label")
        self.add_class("dataset", 2, "lid")
        self.add_class("dataset", 3, "garbage")
        self.add_class("dataset", 4, "clear_bottle")
        self.add_class("dataset", 5, "not_clear_bottle")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in listdir(annotations_dir):
            # extract image id
            image_id = filename[:-4]
            
            img_path = images_dir + image_id + '.jpg'
            ann_path = annotations_dir + filename
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//object'):
        #for box in root.findall('.//bndbox'):
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
 
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if (box[4] == 'label'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('label'))
            elif (box[4] == 'lid'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('lid'))
            elif (box[4] == 'garbage'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('garbage'))
            elif (box[4] == 'clear_bottle'):
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(self.class_names.index('clear_bottle'))
            elif (box[4] == 'not_clear_bottle'):
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(self.class_names.index('not_clear_bottle'))        

            #class_ids.append(self.class_names.index('PET'))
        return masks, asarray(class_ids, dtype='int32')
 
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class PETConfig(Config):
    NAME = "PET_cfg"
    NUM_CLASSES = 1 + 5
    STEPS_PER_EPOCH = 407


 
# define the prediction configuration
class PredictionConfig(Config):
    NAME = "PET_cfg"
    NUM_CLASSES = 1 + 5
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 

 
# load the train dataset
train_set = PETDataset()
train_set.load_dataset('data/train', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = PETDataset()
test_set.load_dataset('data/test', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

def show_annotation(image_id):
    image = train_set.load_image(image_id)
    print(image.shape)
    # load image mask
    mask, class_ids = train_set.load_mask(image_id)
    print(mask.shape)
    # plot image
    pyplot.imshow(image)
    # plot mask
    pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
    pyplot.show()


def train_m():
    config = PETConfig()
    #config.display()

    model = modellib.MaskRCNN(mode='training', model_dir='model_detection/', config=config)
    model.load_weights('model_detection/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')


#--------------------------------------------------------------training
train_m()
#--------------------------------------------------------------



def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        image = dataset.load_image(i+12)
        mask, _ = dataset.load_mask(i+12)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]

        pyplot.subplot(n_images, 2, i*2+1)
        pyplot.imshow(image)
        pyplot.title('Actual')
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        pyplot.subplot(n_images, 2, i*2+2)
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()

        for box in yhat['rois']:
            y1, x1, y2, x2 = box
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
    
    pyplot.show()

#def predict_PET(image, cfg):

#cfg = PredictionConfig()
#model = modellib.MaskRCNN(mode='inference', model_dir='model_detection/', config=cfg)
#model.load_weights('model_detection/mask_rcnn_pet_cfg_0020.h5', by_name=True)

#plot_actual_vs_predicted(test_set, model, cfg)



def evaluate_model(dataset, model, cfg):
    APs = list()
    for i, image_id in enumerate(dataset.image_ids):
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        print(i)
    mAP = mean(APs)
    return mAP

'''
cfg = PredictionConfig()
model = modellib.MaskRCNN(mode='inference', model_dir='model_detection/', config=cfg)
model.load_weights('model_detection/mask_rcnn_pet_cfg_0020.h5', by_name=True)
#train_mAP = evaluate_model(train_set, model, cfg)
#print("Train mAP: %.3f" % train_mAP)
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
'''