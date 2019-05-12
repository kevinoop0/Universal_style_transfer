# The simple implemention of Universal_style_transfer

This is a Pytorch implementation of the "Universal Style Transfer via Feature Trasforms" NIPS17 [paper](https://arxiv.org/abs/1705.08086).

Given a __content image__ and an arbitrary __style image__,
the program attempts to transfer the visual style characteristics extracted from the style image to the content image generating __stylized ouput__.  

The core architecture is a VGG19 Convolutional Autoencoder performing 
Whitening and Coloring Transformation on the content and style features
in the bottleneck layer.   

## Installation
+ Needed Python packages can be installed using [`conda`](https://www.anaconda.com/download/) package manager by running `conda env create -f environment.yaml`



## Usage
`python main.py ARGS`

Possible ARGS are:
+  `--content CONTENT` path of the content image (or a directory containing images) to be trasformed;
+  `--style STYLE` path of the style image (or a directory containing images) to use;
+  `--contentSize CONTENTSIZE` reshape content image to have the new specified maximum size (keeping aspect ratio);
+  `--styleSize STYLESIZE` reshape style image to have the new specified maximum size (keeping aspect ratio);
+  `--outDir OUTDIR` path of the directory where stylized results will be saved (default is `outputs/`);
+  `--alpha ALPHA` hyperparameter balancing the blending between original content features and WCT-transformed features (default is `0.2`);
+  `--no-cuda` flag to enable CPU-only computations (default is `False` i.e. GPU (CUDA) accelaration);

An example

```python
python main.py --content inputs/contents/in4.jpg --style inputs/styles/candy.jpg  
```



## Result



<img src="http://m.qpic.cn/psb?/V12kySKV4IhBFe/nvTrvfFqtVdVol1KvcptErBE7hm0eUA4WZj5Szh7l2Q!/b/dL4AAAAAAAAA&bo=AAYAAgAAAAARBzQ!&rf=viewer_4">