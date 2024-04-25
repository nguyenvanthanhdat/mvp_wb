import gradio as gr
from re_func_tab.func_tab1 import *

def Tab1():
    with gr.Tab("Bright Image") as bright_tab:
        # gr.Markdown("Bright Image")
        with gr.Row():
            u = gr.UploadButton("Please upload your images", file_count="single")
        with gr.Row():
            with gr.Column():
                input = gr.Image(label= "Your picture")
            with gr.Column():
                output = gr.Image(label= "Output")
        with gr.Row():
            wb = gr.Button("White Balance")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr.Markdown("BRIGT UP IMAGE")
                with gr.Row():
                        alpha = gr.Slider(0, 10, step=0.01, label="alpha")
                        beta = gr.Slider(0, 100, step=1, label="beta")
                
                with gr.Row():
                    bright_image = gr.Image(label= "Bright image")
        u.upload(upload_file, u, [u, input, wb])
        wb.click(white_balance, u,[u, output, wb])
        alpha.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
        beta.release(bright_up, inputs=[alpha, beta], outputs=[bright_image])
    return bright_tab
    