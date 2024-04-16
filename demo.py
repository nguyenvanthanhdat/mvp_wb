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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_raw_image(img_path, file_name):
    with open(img_path, 'rb') as f:
        raw_image = rawpy.imread(f)
    rgb = raw_image.postprocess()
    # rgb = cv2.resize(rgb, (256, 256))
    imageio.imsave(os.path.join("gradio_input", "input" + '.jpg'), rgb)
    image = cv2.imread(os.path.join("gradio_input", "input" + '.jpg'))
    resized_image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("gradio_input", "input" + '.jpg'), resized_image)
    return rgb

def white_balance(file_name):  
    os.system(
        """
        python test.py --testing-dir ./gradio_input \
            --outdir ./gradio_output \
            --gpu 0 \
        """
    )
    # return cv2.imread(os.path.join("gradio_output", file_name + '_WB.png'))
    img_wb = Image.open(os.path.join("gradio_output", "input" + '_WB.png'))
    return [gr.UploadButton(visible=True), gr.Image(value=img_wb, label= f"White balance image"), gr.Button(visible=False)]
    
def bright_up(alpha, beta):
    image = cv2.imread(os.path.join("gradio_output", "input" + '_WB.png'))
    new_image = np.zeros(image.shape, image.dtype)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join("gradio_output", "input" + '_WB_new.png'), new_image)
    return Image.open(os.path.join("gradio_output", "input" + '_WB_new.png'))

def upload_file(file_path):
    # print(file_path)
    file_name = Path(file_path).name
    try:
        img = Image.open(file_path)
        img = img.save(os.path.join("gradio_input", "input" + '.jpg'))
        img = Image.open(os.path.join("gradio_input", "input" + '.jpg'))
    except:
        img = convert_raw_image(img_path=file_path, file_name=file_name.split(".")[0])
    # img_wb = white_balance(file_name.split(".")[0])
    return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Button(visible=True)]
    # return [gr.UploadButton(visible=False), gr.Image(value=img, label= f"{file_name}"), gr.Image(value=img_wb, label= "image white balance")]

# def main():
with gr.Blocks() as mvp:
    with gr.Row():
        u = gr.UploadButton("Please upload your images", file_count="single")
    with gr.Row():
        with gr.Column():
            input = gr.Image(label= "Your picture")
        with gr.Column():
            output = gr.Image(label= "Output")
    with gr.Row():
        generate = gr.Button("White Balance")
    u.upload(upload_file, u, [u, input, generate])
    generate.click(white_balance, u,[u, output, generate])
    # generate = gr.Button("Generate")
    with gr.Row():
        with gr.Column():
            alpha = gr.Slider(0, 10, step=0.01)
        with gr.Column():
            beta = gr.Slider(0, 100, step=1)
    with gr.Row():
        bright_image = gr.Image(label= "Bright image")
    alpha.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
    beta.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
    
# mvp.launch()
if __name__ == "__main__":
    os.makedirs("gradio_input", exist_ok=True)
    os.makedirs("gradio_output", exist_ok=True)
    mvp.launch()
    # main()