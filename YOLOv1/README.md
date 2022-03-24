# Notes
* Reference: [YOLOv1 from Scratch](https://www.youtube.com/watch?v=n9_XyCGr-MI)
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
