import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import glob
import random
import os
from matplotlib.patches import Rectangle
from matplotlib import pyplot

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['RBC', 'WBC', 'Platelets']

class PredictionConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "Cell_cfg"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = 4

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=PredictionConfig(),
                             model_dir='./')

# Load the weights into the model.
model.load_weights(filepath="Cell_mask_rcnn_trained.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
all_images= glob.glob(os.path.abspath('./')+'/images/*.jpg')
rand_image = all_images[359]#random.randint(288,300)]
image = cv2.imread(rand_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)
r=r[0]

# Visualize the detected objects.
# mrcnn.visualize.display_instances(image=image, 
#                                   boxes=r['rois'], 
#                                   masks=r['masks'], 
#                                   class_ids=r['class_ids'], 
#                                   class_names=CLASS_NAMES, 
#                                   scores=r['scores'])

pyplot.imshow(image)
ax = pyplot.gca()
class_names = ['RBC', 'WBC', 'Platelets']
R_count=0
W_count=0
P_count=0
class_id_counter=1
for box in r['rois']:
    #print(box)
#get coordinates
    detected_class_id = r['class_ids'][class_id_counter-1]
    #print(detected_class_id)
    #print("Detected class is :", class_names[detected_class_id-1])
    y1, x1, y2, x2 = box
    #calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    #create the shape
    ax.annotate(class_names[detected_class_id-1], (x1, y1), color='black', weight='bold', fontsize=10, ha='center', va='center')
    if class_names[detected_class_id-1]=='RBC':
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        R_count+=1
    elif class_names[detected_class_id-1]=='WBC':
        rect = Rectangle((x1, y1), width, height, fill=False, color='green')
        W_count+=1
    elif class_names[detected_class_id-1]=='Platelets':
        rect = Rectangle((x1, y1), width, height, fill=False, color='blue')
        P_count+=1
        
#draw the box
    ax.add_patch(rect)
    class_id_counter+=1

print("RBCs : {}".format(R_count),
      "WBCs : {}".format(W_count),
      "Platelets : {}".format(P_count), sep="\n")
#show the figure
pyplot.show()


