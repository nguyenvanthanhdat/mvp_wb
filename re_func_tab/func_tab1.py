import gradio as gr
from pathlib import Path
import os
import rawpy
import torch
from src import wb_net
import numpy as np
import cv2
import imageio
from PIL import Image
from matplotlib import pyplot as plt
import sys
from typing import Any, Dict, List

def convert_raw_image(img_path, file_name):
    with open(img_path, 'rb') as f:
        raw_image = rawpy.imread(f)
    rgb = raw_image.postprocess()
    # rgb = cv2.resize(rgb, (256, 256))
    imageio.imsave(os.path.join("outputs", "input_tab1", "input" + '.jpg'), rgb)
    image = cv2.imread(os.path.join("outputs", "input_tab1", "input" + '.jpg'))
    resized_image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("outputs", "input_tab1", "input" + '.jpg'), resized_image)
    return rgb

def white_balance(file_name):  
    os.system(
        """
        python test.py --testing-dir ./outputs/input_tab1 \
            --outdir ./outputs/output_tab1 \
            --gpu 0 \
        """
    )
    # return cv2.imread(os.path.join("wb_output", file_name + '_WB.png'))
    img_wb = Image.open(os.path.join("outputs", "output_tab1", "input" + '_WB.png'))
    return [gr.UploadButton(visible=True), gr.Image(value=img_wb, label= f"White balance image"), gr.Button(visible=False)]
    
def bright_up(alpha, beta):
    image = cv2.imread(os.path.join("outputs", "output_tab1", "input" + '_WB.png'))
    new_image = np.zeros(image.shape, image.dtype)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join("outputs", "output_tab1", "input" + '_WB_new.png'), new_image)
    new_image = Image.open(os.path.join("outputs", "output_tab1", "input" + '_WB_new.png'))
    width, height = new_image.size
    # print(width, height)
    # return [new_image, gr.Slider(0, width, step=1, visible=True), gr.Slider(0, height, step=1, visible=True)]
    return new_image

def upload_file(file_path):
    # print(file_path)
    file_name = Path(file_path).name
    try:
        img = Image.open(file_path)
        img = img.save(os.path.join("input_tab1", "input" + '.jpg'))
        img = Image.open(os.path.join("input_tab1", "input" + '.jpg'))
    except:
        img = convert_raw_image(img_path=file_path, file_name=file_name.split(".")[0])
    # img_wb = white_balance(file_name.split(".")[0])
    return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Button(visible=True)]
    # return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Image(value=img_wb, label= "image white balance")]
