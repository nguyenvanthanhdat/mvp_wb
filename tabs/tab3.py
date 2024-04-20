import gradio as gr
from func_tab.func_tab3 import *

def Tab3():
    with gr.Tab("Insert Items") as insert_tab:
        # gr.Markdown("Bright Image")
        # with gr.Column():
        #     with gr.Row():
        #         gr.Markdown("BRIGT UP IMAGE")
        #     with gr.Row():
        #             alpha = gr.Slider(0, 10, step=0.01)
        #             beta = gr.Slider(0, 100, step=1)
            
        #     with gr.Row():
        #         bright_image = gr.Image(label= "Bright image")
        with gr.Column():
            with gr.Row():
                gr.Markdown("INSERT THINGS (Test with (2090, 560))")
            with gr.Row():
                insert_x = gr.Slider(visible=True)
                insert_y = gr.Slider(visible=True)
            with gr.Row():
                image_point = gr.Image(label= "Choose item")# Inserted image")
            # with gr.Row():
                
    
        with gr.Column():
            with gr.Row():
                prompt = gr.Text(label="Prompt")
            with gr.Row():
                insert_button = gr.Button("Insert item")
        with gr.Row():
            with gr.Column():
                image_option_1 = gr.Image(label= "Option 1")
                choose1 = gr.Button("Choose 1")
            with gr.Column():
                image_option_2 = gr.Image(label= "Option 2")
                choose2 = gr.Button("Choose 2")
            with gr.Column():      
                image_option_3 = gr.Image(label= "Option 3")
                choose3 = gr.Button("Choose 3")
        with gr.Row():
            image_insertd = gr.Image(visible=False)
            
    insert_x.release(get_point, inputs=[insert_x, insert_y], 
                     outputs=[image_point, insert_x, insert_y]
    )
    insert_y.release(get_point, inputs=[insert_x, insert_y], 
                     outputs=[image_point, insert_x, insert_y]
    )
    insert_button.click(insert_item, inputs=[image_point, prompt, insert_x, insert_y], 
                    outputs=[image_option_1, image_option_2, image_option_3]
    )
    choose1.click(choose, image_option_1, outputs=[image_option_1, image_option_2, image_option_3, image_insertd,
                                   choose1, choose2, choose3])
    choose2.click(choose, image_option_2, outputs=[image_option_1, image_option_2, image_option_3, image_insertd,
                                   choose1, choose2, choose3])
    choose3.click(choose, image_option_3, outputs=[image_option_1, image_option_2, image_option_3, image_insertd,
                                   choose1, choose2, choose3])
    return insert_tab