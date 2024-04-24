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


with gr.Blocks() as mvp:
    Tab1()
    Tab2()
    Tab3()

if __name__ == "__main__":
    os.makedirs("wb_input", exist_ok=True)
    os.makedirs("wb_output", exist_ok=True)
    os.makedirs("remove_input", exist_ok=True)
    os.makedirs("insert_input", exist_ok=True)
    mvp.launch(share=True)
    # main()