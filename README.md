# Fast Style Transfer in [TensorFlow](https://github.com/tensorflow/tensorflow)

Stylize your images...

# Implementation Details
This implementation uses TensorFlow to train a fast style transfer network.
The basic code uses the same transformation network as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output `tanh` layer is slightly different. 
The loss function used is close to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation (e.g. we use `relu1_1` rather than `relu1_2`). 
Empirically, this results in larger scale style features in transformations.

# API Documentation
## style.py 

`style.py` trains networks that can transfer styles from artwork into images.

**Flags**
- `--checkpoint-dir`: Directory to save checkpoint in. Required.
- `--style`: Path to style image. Required.
- `--train-path`: Path to training images folder. Default: `data/train2014`.
- `--test`: Path to content image to test network on at at every checkpoint iteration. Default: no image.
- `--test-dir`: Path to directory to save test images in. Required if `--test` is passed a value.
- `--epochs`: Epochs to train for. Default: `2`.
- `--batch_size`: Batch size for training. Default: `4`.
- `--checkpoint-iterations`: Number of iterations to go for between checkpoints. Default: `2000`.
- `--vgg-path`: Path to VGG19 network (default). Can pass VGG16 if you want to try out other loss functions. Default: `data/imagenet-vgg-verydeep-19.mat`.
- `--content-weight`: Weight of content in loss function. Default: `7.5e0`.
- `--style-weight`: Weight of style in loss function. Default: `1e2`.
- `--tv-weight`: Weight of total variation term in loss function. Default: `2e2`.
- `--learning-rate`: Learning rate for optimizer. Default: `1e-3`.
- `--slow`: For debugging loss function. Direct optimization on pixels using Gatys' approach. Uses `test` image as content value, `test_dir` for saving fully optimized images.

### Training Style Transfer Networks
Use `style.py` to train a new style transfer network. Run `python style.py` to view all the possible parameters. Training takes 4-6 hours on a Maxwell Titan X. **Before you run this, you should run `setup.sh`**. Example usage:

    python style.py --style my/path/style_img.jpg \
      --checkpoint-dir checkpoint/path \
      --test my/path/test_img.jpg \
      --test-dir my/path/test/dir \
      --content-weight 1.5e1 \
      --checkpoint-iterations 1000 \
      --batch-size 8

## evaluate.py
`evaluate.py` evaluates trained networks given a checkpoint directory. If evaluating images from a directory, every image in the directory must have the same dimensions.

**Flags**
- `--checkpoint`: Directory or `ckpt` file to load checkpoint from. Required.
- `--in-path`: Path of image or directory of images to transform. Required.
- `--out-path`: Out path of transformed image or out directory to put transformed images from in directory (if `in_path` is a directory). Required.
- `--device`: Device used to transform image. Default: `/cpu:0`.
- `--batch-size`: Batch size used to evaluate images. In particular meant for directory transformations. Default: `4`.
- `--allow-different-dimensions`: Allow different image dimensions. Default: not enabled

### Evaluating Style Transfer Networks
Use `evaluate.py` to evaluate a style transfer network. Run `python evaluate.py` to view all the possible parameters. Evaluation takes 100 ms per frame (when batch size is 1) on a Maxwell Titan X. Example usage:

    python evaluate.py --checkpoint path/to/style/model.ckpt \
      --in-path dir/of/test/imgs/ \
      --out-path dir/for/results/
