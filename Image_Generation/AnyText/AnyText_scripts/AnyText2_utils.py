import cv2
import numpy as np
import torch

def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image

def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    # if max_dimension > max_length:
    scale_factor = max_length / max_dimension
    new_width = int(round(width * scale_factor))
    new_height = int(round(height * scale_factor))
    new_size = (new_width, new_height)
    img = cv2.resize(img, new_size)
    
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img

def separate_pos_imgs(img, sort_priority, gap=102):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        component = np.zeros_like(img)
        component[labels == label] = 255
        components.append((component, centroids[label]))
    if sort_priority == '↕':
        fir, sec = 1, 0  # top-down first
    elif sort_priority == '↔':
        fir, sec = 0, 1  # left-right first
    components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
    sorted_components = [c[0] for c in components]
    return sorted_components

def find_polygon(image, min_rect=False):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
    if min_rect:
        # get minimum enclosing rectangle
        rect = cv2.minAreaRect(max_contour)
        poly = np.int0(cv2.boxPoints(rect))
    else:
        # get approximate polygon
        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        poly = cv2.approxPolyDP(max_contour, epsilon, True)
        n, _, xy = poly.shape
        poly = poly.reshape(n, xy)
    cv2.drawContours(np.ascontiguousarray(image, dtype=np.uint8), [poly], -1, 255, -1)
    return poly, image

def arr2tensor(arr, bs, device, dtype):
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().to(device, dtype) #cuda(0)
    
    # if self.use_fp16:
    #     _arr = _arr.half()
        
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr