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
from tabs.tab1 import Tab1
from tabs.tab2 import Tab2
from tabs.tab3 import Tab3

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def convert_raw_image(img_path, file_name):
#     with open(img_path, 'rb') as f:
#         raw_image = rawpy.imread(f)
#     rgb = raw_image.postprocess()
#     # rgb = cv2.resize(rgb, (256, 256))
#     imageio.imsave(os.path.join("wb_input", "input" + '.jpg'), rgb)
#     image = cv2.imread(os.path.join("wb_input", "input" + '.jpg'))
#     resized_image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_LINEAR)
#     cv2.imwrite(os.path.join("wb_input", "input" + '.jpg'), resized_image)
#     return rgb

# def white_balance(file_name):  
#     os.system(
#         """
#         python test.py --testing-dir ./wb_input \
#             --outdir ./wb_output \
#             --gpu 0 \
#         """
#     )
#     # return cv2.imread(os.path.join("wb_output", file_name + '_WB.png'))
#     img_wb = Image.open(os.path.join("wb_output", "input" + '_WB.png'))
#     return [gr.UploadButton(visible=True), gr.Image(value=img_wb, label= f"White balance image"), gr.Button(visible=False)]
    
# def bright_up(alpha, beta):
#     image = cv2.imread(os.path.join("wb_output", "input" + '_WB.png'))
#     new_image = np.zeros(image.shape, image.dtype)
#     # for y in range(image.shape[0]):
#     #     for x in range(image.shape[1]):
#     #         for c in range(image.shape[2]):
#     #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
#     new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
#     cv2.imwrite(os.path.join("wb_output", "input" + '_WB_new.png'), new_image)
#     new_image = Image.open(os.path.join("wb_output", "input" + '_WB_new.png'))
#     width, height = new_image.size
#     # print(width, height)
#     # return [new_image, gr.Slider(0, width, step=1, visible=True), gr.Slider(0, height, step=1, visible=True)]
#     return new_image

# def upload_file(file_path):
#     # print(file_path)
#     file_name = Path(file_path).name
#     try:
#         img = Image.open(file_path)
#         img = img.save(os.path.join("wb_input", "input" + '.jpg'))
#         img = Image.open(os.path.join("wb_input", "input" + '.jpg'))
#     except:
#         img = convert_raw_image(img_path=file_path, file_name=file_name.split(".")[0])
#     # img_wb = white_balance(file_name.split(".")[0])
#     return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Button(visible=True)]
#     # return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Image(value=img_wb, label= "image white balance")]

# def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
#     coords = np.array(coords)
#     labels = np.array(labels)
#     color_table = {0: 'red', 1: 'green'}
#     for label_value, color in color_table.items():
#         points = coords[labels == label_value]
#         ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
#                    s=size, edgecolor='white', linewidth=1.25)

# def get_point(remove_x, remove_y):
#     img = Image.open(os.path.join("wb_output", "input_WB_new" + '.png'))
#     dpi = plt.rcParams['figure.dpi']
#     height, width = np.array(img).shape[:2]
#     plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
#     plt.imshow(img)
#     plt.axis('off')
#     show_points(plt.gca(), [remove_x, remove_y], 1,
#                 size=(width*0.04)**2)
#     plt.savefig(os.path.join("remove_input", "with_point" + '.png'), bbox_inches='tight', pad_inches=0)
#     plt.close()
#     point_image = Image.open(os.path.join("remove_input", "with_point" + '.png'))
#     return point_image, gr.Slider(minimum=0, maximum=width - 1, step=1), gr.Slider(minimum=0, maximum=height - 1, step=1)
    

# def remove_item(file_path):
#     pass

# def main():
with gr.Blocks() as mvp:
    Tab1()
    Tab2()
    Tab3()
    # with gr.Tab("Bright Image"):
    #     gr.Markdown("Bright Image")
    # with gr.Tab("Remove Item"):
    #     gr.Markdown("Remove Item") 
    # with gr.Row():
    #     u = gr.UploadButton("Please upload your images", file_count="single")
    # with gr.Row():
    #     with gr.Column():
    #         input = gr.Image(label= "Your picture")
    #     with gr.Column():
    #         output = gr.Image(label= "Output")
    # with gr.Row():
    #     wb = gr.Button("White Balance")

    # # generate = gr.Button("Generate")
    # with gr.Row():
    #     with gr.Column():
    #         with gr.Row():
    #             gr.Markdown("BRIGT UP IMAGE")
    #         with gr.Row():
    #                 alpha = gr.Slider(0, 10, step=0.01)
    #                 beta = gr.Slider(0, 100, step=1)
            
    #         with gr.Row():
    #             bright_image = gr.Image(label= "Bright image")
    #     with gr.Column():
    #         with gr.Row():
    #             gr.Markdown("REMOVE THINGS (Test with (2090, 560))")
    #         with gr.Row():
    #             remove_x = gr.Slider(visible=True)
    #             remove_y = gr.Slider(visible=True)
    #         with gr.Row():
    #             image_point = gr.Image(label= "Removed image")
    
    #     with gr.Column():
    #         with gr.Row():
    #             rm_button = gr.Button("Remove item")
    
    # # u.upload(upload_file, u, [u, input, wb])
    # # wb.click(white_balance, u,[u, output, wb])
    # # alpha.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
    # # beta.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
    # remove_x.release(get_point, inputs=[remove_x, remove_y], outputs=[image_point, remove_x, remove_y])
    # remove_y.release(get_point, inputs=[remove_x, remove_y], outputs=[image_point, remove_x, remove_y])
    # rm_button.click()
    
# mvp.launch()
if __name__ == "__main__":
    os.makedirs("wb_input", exist_ok=True)
    os.makedirs("wb_output", exist_ok=True)
    os.makedirs("remove_input", exist_ok=True)
    os.makedirs("insert_input", exist_ok=True)
    mvp.launch()
    # main()