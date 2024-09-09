import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np

import onnxruntime as ort
from transformers import MaskFormerFeatureExtractor

from image_utils import convert_semantic_segmentation, SEGMENT_COLOR


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL



class InteriorSegmentONNX:
    def __init__(self, model_name_or_path) -> None:
        self.load_model(model_name_or_path)
        
        # load label
        with open(os.path.join(model_name_or_path, 'config.json'), 'r') as file:
            config_file = json.load(file)

        self.label = config_file['label2id']
    
    def load_model(self, model_name_or_path, provider=["CUDAExecutionProvider","CPUExecutionProvider"]):
        """Load model from hub

        Args:
            model_name_or_path (str): model path
            provider (list, optional): Provider run onnx model
        """
        _model_name_or_path = os.path.join(model_name_or_path, 'model.onnx')
        self.model = ort.InferenceSession(
            _model_name_or_path,
            sess_options=sess_options,
            providers=provider
        )
        
        self.inputs = [_inp.name for _inp in self.model.get_inputs()]
        self.outputs = [_opt.name for _opt in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
        
        self.processor = MaskFormerFeatureExtractor.from_pretrained(model_name_or_path)
        
    def inferernence(self, img, cls_name):
        """ Prediction image

        Args:
            img (np.array): An image
            cls_name (list): name of segment want to get

        Returns:
            image(np.array): Image with segmentation
        """
        # preprocess feature extraction
        input_tensor = self.preprocess(img)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # model predict
        outputs = self.model.run(self.outputs, dict(zip(self.inputs, input_tensor)))
        semantic_segment = convert_semantic_segmentation(
            class_queries=outputs[0], 
            mask_queries=outputs[1], 
            target_sizes=[img.size[::-1]]
        )[0]
        segmented = self.postprocess(semantic_segment, cls_name)
        
        return segmented
    
    def preprocess(self, img):
        """Prepare input model. Convert Image to tensor (np.array)

        Args:
            img (np.array): An image 

        Returns:
            pixel_values (np.array): An numpy array
        """
        img = img.convert('RGB')
        input_tensor = self.processor(img, return_tensors='np')
        
        return input_tensor['pixel_values']
    
    def postprocess(self, segment, name):
        """Post processing get segmentation mask

        Args:
            segment (np.array): An image mask with shape 2D
            name (list): name of segment

        Returns:
            Image(np.array): Image with segment mask
        """
        color_seg = np.zeros((segment.shape[0], segment.shape[1], 3), dtype=np.uint8)
        for _name in name:
            assert _name in list(self.label.keys()), f'Could not found {name} in data'
            color_seg[segment.cpu().numpy() == self.label[_name], :] = SEGMENT_COLOR[_name]
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        img_mask = color_seg.astype(np.uint8)
    
        return img_mask
    
    
if __name__ == '__main__':
    from PIL import Image
    
    model_onnx = InteriorSegmentONNX('weights/semantic_segmentation/1')
    image = Image.open('datahub/1394_G.png').convert('RGB')
    name_attr = ['ceiling', 'wall', 'floor']
    
    input_tensort = model_onnx.inferernence(image, cls_name=name_attr)
    print("\ninput_tensort: ", input_tensort.shape)
    
    img_predict = Image.fromarray(input_tensort)
    img_predict.save('test.jpg')