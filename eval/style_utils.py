import os
import cv2
import math
import json
import numpy as np
import pygame, pygame.locals
from omegaconf import OmegaConf

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size //
                       2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in [
               '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


class devdata(Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.imageFiles = [os.path.join(dataRoot, filename) for filename 
                           in os.listdir(dataRoot) if CheckImageFile(filename)]
        self.gtFiles = [os.path.join(gtRoot, filename) for filename 
                        in os.listdir(dataRoot) if CheckImageFile(filename)]
        self.loadSize = loadSize

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        gt = Image.open(self.gtFiles[index])
        to_scale = gt.size
        # to_scale = self.loadSize
        inputImage = ImageTransform(to_scale)(img.convert('RGB'))
        groundTruth = ImageTransform(to_scale)(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth, path

    def __len__(self):
        return len(self.imageFiles)


def center2size(surf, size):

    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    canvas[(size_h-surf_h)//2:(size_h-surf_h)//2+surf_h, (size_w-surf_w)//2:(size_w-surf_w)//2+surf_w] = surf
    return canvas

def render_normal(font, text):
        
    # get the number of lines
    lines = text.split('\n')
    lengths = [len(l) for l in lines]

    # font parameters:
    line_spacing = font.get_sized_height() + 1

    # initialize the surface to proper size:
    line_bounds = font.get_rect(lines[np.argmax(lengths)])
    fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing * len(lines)))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    bbs = []
    space = font.get_rect('O')
    x, y = 0, 0
    for l in lines:
        x = 0 # carriage-return
        y += line_spacing # line-feed

        for ch in l: # render each character
            if ch.isspace(): # just shift
                x += space.width
            else:
                # render the character
                ch_bounds = font.render_to(surf, (x,y), ch)
                ch_bounds.x = x + ch_bounds.x
                ch_bounds.y = y - ch_bounds.y
                x += ch_bounds.width
                bbs.append(np.array(ch_bounds))

    bbs = np.array(bbs)
    
    alpha = pygame.surfarray.pixels_alpha(surf)
    surf_arr = alpha.swapaxes(0,1)               
    
    return surf_arr, bbs

def center_warpPerspective(img, H, center, size):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size,
                    cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return img

def center_pointsPerspective(points, H, center):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    return M.dot(points)

def perspective(img, rotate_angle, zoom, shear_angle, perspect, pad): # w first

    rotate_angle = rotate_angle * math.pi / 180.
    shear_x_angle = shear_angle[0] * math.pi / 180.
    shear_y_angle = shear_angle[1] * math.pi / 180.
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect
    
    H_scale = np.array([[scale_w, 0, 0],
                        [0, scale_h, 0],
                        [0, 0, 1]], dtype = np.float32)
    H_rotate = np.array([[math.cos(rotate_angle), math.sin(rotate_angle), 0],
                         [-math.sin(rotate_angle), math.cos(rotate_angle), 0],
                         [0, 0, 1]], dtype = np.float32)
    H_shear = np.array([[1, math.tan(shear_x_angle), 0],
                        [math.tan(shear_y_angle), 1, 0], 
                        [0, 0, 1]], dtype = np.float32)
    H_perspect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [perspect_x, perspect_y, 1]], dtype = np.float32)

    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)
    points = np.ones((3, 4), dtype = np.float32)
    points[:2, 0] = np.array([0, 0], dtype = np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype = np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype = np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype = np.float32).T
    perspected_points = center_pointsPerspective(points, H, img_center)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    canvas_w = int(2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]), 
                      np.max(perspected_points[0, :]) - img_center[0])) + 10
    canvas_h = int(2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]), 
                      np.max(perspected_points[1, :]) - img_center[1])) + 10
    
    canvas = np.zeros((canvas_h, canvas_w), dtype = np.uint8)
    tly = (canvas_h - img_h) // 2
    tlx = (canvas_w - img_w) // 2
    canvas[tly:tly+img_h, tlx:tlx+img_w] = img
    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    canvas = center_warpPerspective(canvas, H, canvas_center, canvas_size)
    loc = np.where(canvas > 127)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    text_w = maxx - minx + 1
    text_h = maxy - miny + 1
    resimg = np.zeros((text_h + pad[2] + pad[3], text_w + pad[0] + pad[1])).astype(np.uint8)
    resimg[pad[2]:pad[2]+text_h, pad[0]:pad[0]+text_w] = canvas[miny:maxy+1, minx:maxx+1]
    return resimg


def load_config(config_name):
    config_path = os.path.join("configs", f"{config_name}.yaml")
    return OmegaConf.load(config_path)

def update_results(results_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)