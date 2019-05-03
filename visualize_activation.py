import cv2
import numpy as np
from vis.visualization import visualize_cam


def overlay_cam(model, img, prediction, layer_idx, last_conv_layer_id, cutoff=30):

    heat_map = visualize_cam(model, layer_idx, np.argmax(prediction), img, last_conv_layer_id)
    
    ## zero all the regions that are below a threshold
    heat_map[heat_map[:,:,0]<cutoff] = 0

    # convert normalized image to 0-255 range,
    # multiplying with 330 instead of 255 to increase the brightness of the images
    img -= np.min(img)
    img = img / np.max(img) * 330.0
    img[img > 255.0] = 255.0
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)

    overlayed_im = cv2.addWeighted(heat_map,0.4,img,0.6,0)
    overlayed_im = cv2.cvtColor(overlayed_im, cv2.COLOR_RGB2BGR)

    return overlayed_im