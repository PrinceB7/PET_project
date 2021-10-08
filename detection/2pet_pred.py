from os import listdir
from numpy import expand_dims
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import mold_image
from Mask_RCNN.mrcnn import model as modellib
import cv2 as cv
 
class PredictionConfig(Config):
    NAME = "tumor_cfg"
    NUM_CLASSES = 1 + 5
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def predict_(foldername, model, cfg):
    filenames = [foldername + x for x in listdir(foldername)]
    for i, filename in enumerate(filenames):
        print(i)
        image = cv.imread(filename)
        image = cv.resize(image, (900,1200), interpolation = cv.INTER_AREA)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        pred = model.detect(sample, verbose=0)[0]
        
        for i, box in enumerate(pred['rois']):
            y1, x1, y2, x2 = box
            #print(name) bgr
            if pred["class_ids"][i] == 1:
                color = (0,0,255)
            elif pred["class_ids"][i] == 2:
                color = (255,0,0)
            elif pred["class_ids"][i] == 3:
                color = (0,255,0)   
            elif pred["class_ids"][i] == 4:
                color = (0,255,255)     
            elif pred["class_ids"][i] == 5:
                color = (255,0,255)    
            image = cv.rectangle(image, (x1,y1), (x2, y2), color, 3)
        cv.imwrite(filename, image)

cfg = PredictionConfig()
model = modellib.MaskRCNN(mode='inference', model_dir='model_detection/', config=cfg)
model.load_weights('model_detection/mask_rcnn_pet_cfg_0030.h5', by_name=True)

predict_('test_new_trash/', model, cfg)
print('\ndone')
#plot_actual_vs_predicted(test_set, model, cfg)