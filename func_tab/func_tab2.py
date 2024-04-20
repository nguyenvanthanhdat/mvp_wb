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

def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

def get_point(remove_x, remove_y):
    img = Image.open(os.path.join("wb_output", "input_WB_new" + '.png'))
    dpi = plt.rcParams['figure.dpi']
    height, width = np.array(img).shape[:2]
    plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), [remove_x, remove_y], 1,
                size=(width*0.04)**2)
    plt.savefig(os.path.join("remove_input", "with_point" + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    point_image = Image.open(os.path.join("remove_input", "with_point" + '.png'))
    return point_image, gr.Slider(minimum=0, maximum=width - 1, step=1), gr.Slider(minimum=0, maximum=height - 1, step=1)

def remove_item(file_path, remove_x, remove_y):
    os.system(
        f"""
        python Inpaint-Anything/remove_anything.py \
            --input_img wb_output/input_WB_new.png \
            --coords_type key_in \
            --point_coords {remove_x} {remove_y} \
            --point_labels 1 \
            --dilate_kernel_size 15 \
            --output_dir ./remove_output \
            --sam_model_type "vit_t" \
            --sam_ckpt Inpaint-Anything/weights/mobile_sam.pt \
            --lama_config Inpaint-Anything/lama/configs/prediction/default.yaml \
            --lama_ckpt Inpaint-Anything/pretrained_models/big-lama
        """
    )
    img1 = Image.open(os.path.join("remove_output", "input_WB_new", 'inpainted_with_mask_0.png'))
    img2 = Image.open(os.path.join("remove_output", "input_WB_new", 'inpainted_with_mask_1.png'))   
    img3 = Image.open(os.path.join("remove_output", "input_WB_new", 'inpainted_with_mask_2.png'))

    return [
        gr.Image(value=img1, label= f"Option 1"), 
        gr.Image(value=img2, label= f"Option 2"), 
        gr.Image(value=img3, label= f"Option 3")
    ]
    
def choose(option):
    option = Image.fromarray(option, "RGB")
    option.save(os.path.join("remove_output", "remove_output.png"))
    option = gr.Image(value=option, visible=True)
    return [
        gr.Image(visible=False),
        gr.Image(visible=False),
        gr.Image(visible=False),
        option,
        gr.Button(visible=False),
        gr.Button(visible=False),
        gr.Button(visible=False),
    ]