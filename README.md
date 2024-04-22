# mvp_wb

This is a mvp for cwcc project

### simple run gradio

1. `pip install -r requirements.txt`
2. `python demo.py`

### Changing the contrast and brightness of an image

The formula $g(x) = α * f(x) + β$ 

- Start by increasing the alpha first, and halt adjustments if the color becomes unsuitable and similar to beta.
- The $α$ control the constract, if adjusted too high, this coefficient may lead to color inversion, where light colors become dark and vice versa.
- The $g(x) = α * f(x) + β$, if adjusted too high, this factor can result in overexposure of the photo.
