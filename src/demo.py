import os
import gradio as gr

from modeling.maskformer import InteriorSegment


MODEL = InteriorSegment('weights/maskformer-swin-base-ade')
NAME_ATTR = ['ceiling', 'wall', 'floor']


def get_images(image):
    result = MODEL.inference(image, NAME_ATTR)
    return result[0], result[1], result[2]

demo = gr.Interface(
    fn = get_images,
    inputs=[gr.Image(type="pil", label="Input images")],
    outputs=[gr.Image(label="celling"), gr.Image(label="wall"), gr.Image(label="floor")],
    # examples=[
    #     os.path.join(os.path.dirname(__file__), "datahub/1560_G.png"),
    #     os.path.join(os.path.dirname(__file__), "datahub/256_G.png"),
    #     os.path.join(os.path.dirname(__file__), "datahub/1364_G.png"),
    #     os.path.join(os.path.dirname(__file__), "datahub/1394_G.png"),
    #     os.path.join(os.path.dirname(__file__), "datahub/1505_G.png"),
    # ]
)



if __name__ == "__main__":
    demo.launch(share=True)