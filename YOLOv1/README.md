# Reference: 
* [YOLOv1 from Scratch](https://www.youtube.com/watch?v=n9_XyCGr-MI)
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

# Env
* Python 3.8.12: `conda create -n "YOLOv1" python=3.8`
* Pytorch 11.0: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`


# Information
* Tool: Pytroch
* Dataset: PASCAL VOC Dataset, 20 classes

# Outline
* Idea behind algorithm
* Model architecture
* Loss Function

# YOLO algorithm
* Split an image into `S x S` grids.
* **Each cell** will output a **prediction** with a corresponding **bounding box**.
* The grid which is responsible contains the objects's midpoint.
* Each of the grid have at the top left corner is `(0 0)` and the bottom right is going to be `(1 1)`.
* Each **output** and **label** will be relative to the cell!
* Each bounding box for each grid will have: `[x, y, w, h]` where `x, y` are the coordinates for object midpoint in the grid and **the values should be between 0 and 1**, and `w, h` are the values for width and height of the object and **the values could be greater than 1, if object is taller or wider than a grid.**

## How the labels will actually look
`label_grid = [C1, C2, ..., C20, Pc, x, y, w, h]`, where:
* `C1 ... C20`: 20 different classes
* `Pc`: Probability that there is an object (1 or 0)
* `x, y, w, h`: bounding box

Note: Predictions will look very smiliar to the labels, but we will output two bounding boxes. Because we hope that they will specialize to output different bounding boxes (wide vs tall)

## How the predictions will actually look

`pred_grid = [C1, C2, ..., Pc1, x, y, w, h, Pc2, x, y, w, h]`, where
* `C1 ... C20`: 20 different classes
* `Pc1 , Pc2`: Probability that there is an object
* `x, y, w, h`: bounding box 1 and 2

Limitation: Note: A grid can only detect **ONE** object.

## Remember this is for every cell
* Target shape for one image: `(S, S, 25)`
* Prediction shape for one image: `(S, S, 30)`

# Model Architecture
* Spec for Input, Feature Map, and Output: `(Width, Height, Channel)`
* Spec for Conv. Layer: `(Kernel_size, Kernel_size, Number_of_Kernel)-s-(Stride_size)`
* Spec for Maxpool Layer: `(Pool_size, Pool_size)-s-(Stride_size)`
* Input: `(448, 448, 3)`
    * Conv. Layer: 7x7x64-s-2
    * Maxpool Layer: 2x2-s-2 
* Feature Map: `(112, 112, 192)`
    * Conv. Layer: 3x3x192
    * Maxpool Layer: 2x2-s-2
* Feature Map: `(56, 56, 256)`
    * Conv. Layer: 1x1x28, 3x3x256, 1x1x256, 3x3x512
    * Maxpool Layer: 2x2-s-2
* Feature Map: `(28, 28, 512)`
    * Conv. Layer: {1x1x256, 3x3x512}x4, 1x1x512, 3x3x1024
    * Maxpool Layer: 2x2-s-2
* Feature Map: `(14, 14, 1024)`
    * Conv. Layer: {1x1x512, 3x3x1024}x2, 3x3x1024, 3x3x1024-s-2
* Feature Map: `(7, 7, 1024)`
    * Conv. Layer: 3x3x1024, 3x3x1024
* Feature Map: `(7, 7, 1024)`
    * Conn. Layer
* Feature Map: `(4096)`
    * Conn. Layer
* Output: `(7, 7, 30)`

# Loss Function
* Sum of Squared Errors
* First loss term
    * Box coordinates for mid points
    * Identity function. Be 1, if there's a box in cell i and bbox j is responsible for outputing that bbox. By responsible, it means the highest IOU of any predictor in that grid cell.
* Second Loss Term
    * Width and height
    * Use squared root because we want to treat smaller bbox the same as larger bbox.
* Third Loss Term
    * Possibilities of objects
* Forth Loss Term
    * Which is responsible?
    * The video maker thinks use both of the bbox.
* Fifth Loss Term
    * Classes
    * Instead of using MSE, rather using Cross Entropy Error