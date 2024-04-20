import gradio as gr
from func_tab.func_tab2 import *

def Tab2():
    with gr.Tab("Remove Items") as remove_tab:
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
                gr.Markdown("REMOVE THINGS (Test with (2090, 560))")
            with gr.Row():
                remove_x = gr.Slider(visible=True)
                remove_y = gr.Slider(visible=True)
            with gr.Row():
                image_point = gr.Image(label= "Choose item")# Removed image")
            # with gr.Row():
                
    
        with gr.Column():
            with gr.Row():
                rm_button = gr.Button("Remove item")
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
            image_removed = gr.Image(visible=False)
            
    remove_x.release(get_point, inputs=[remove_x, remove_y], 
                     outputs=[image_point, remove_x, remove_y]
    )
    remove_y.release(get_point, inputs=[remove_x, remove_y], 
                     outputs=[image_point, remove_x, remove_y]
    )
    rm_button.click(remove_item, inputs=[image_point, remove_x, remove_y], 
                    outputs=[image_option_1, image_option_2, image_option_3]
    )
    choose1.click(choose, image_option_1, outputs=[image_option_1, image_option_2, image_option_3, image_removed,
                                   choose1, choose2, choose3])
    choose2.click(choose, image_option_2, outputs=[image_option_1, image_option_2, image_option_3, image_removed,
                                   choose1, choose2, choose3])
    choose3.click(choose, image_option_3, outputs=[image_option_1, image_option_2, image_option_3, image_removed,
                                   choose1, choose2, choose3])
    return remove_tab