import os
import cv2
import numpy as np
from vis.visualization import visualize_cam
from data_generators import get_labels

def save_activation_maps(model, x_test, y_test, log_dir):
    labels = get_labels()
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=-1)
    y_test = np.argmax(y_test, axis=-1)
    incorrect = np.nonzero(predictions != y_test)[0]
    correct = np.nonzero(predictions == y_test)[0]
    for i in incorrect[:20]:
        overlayed_img = overlay_cam(model, x_test[i], predictions[i], -1, -7)
        cv2.imwrite(os.path.join(log_dir, 'incorrect_%d_%s_%s.jpg' % (i, labels[y_test[i]], labels[predictions[i]])),
                    overlayed_img)

    for i in correct[:20]:
        overlayed_img = overlay_cam(model, x_test[i], predictions[i], -1, -7)
        cv2.imwrite(os.path.join(log_dir, 'correct_%d_%s_%s.jpg' % (i, labels[y_test[i]], labels[predictions[i]])),
                    overlayed_img)

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