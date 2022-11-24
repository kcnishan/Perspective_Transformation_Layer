# Perspective_Transformation_Layer
Code and experiments for perspective transformation Layer


## Introduction
The perspective transformation transformation layer can learn adjustable number of multiple viewpoints(homography).

![multi-view](https://user-images.githubusercontent.com/16822926/203680319-046e1141-51f0-4a7e-98e8-ae2f8f34df95.png)



## How to use perspective transformation layer

```
from pers_layer impoort *
from utils import insert_layer

updated_model = insert_layer(0,4,my_model=model)
```
Note: Here insert layer will insert PT layer(s) at the specified position. 
In the above code, the first argument is the position to insert PT layer in the model; second argument is the number of transformation matrix and third 
argument is the model we want to insert PT layer.

## Some additional Application(apart from paper)
Analogous to the state-of-the-art methods, the proposed PT layer has potential in applications of spatial attention and weak supervision. The below figure shows one of many examples in the datasets. We explain the potential using a PTL-1’s outputs of an example input image with two digits. When recognizing each digit on this two-digit image, the PT layer can learn to concentrate on the correct target digit with a viewpoint. This concentration outcome can facilitate spatial attention where we need to learn to highlight certain parts of images to help downstream tasks. Furthermore, the PT layer has the potential to assist weak supervision with inexact labels: when only providing classification labels to a deep learning model, the PT layer could also learn to centralize and localize the to-be-classified object, thus learning the object location on the images with only classification labels.


![attention (2)](https://user-images.githubusercontent.com/16822926/203695781-ee813747-8e24-48a5-b919-41b9b28c5c42.png)
Figure: An example visual attention. (a) The input image containing two digits;(b) and (c) The outputs of a PTL-1 when classifying it as ”3” and ”6” respectively.

## Segmentation
Image segmentation is a domain of computer vision in which a image is divided into segments like objects or part of objects comprising sets of pixels.A semantic segmentation classifies all the pixels of image into classes of objects. Apart from above described experiments, we explored the advantage of our layer for semantic segmentation task.

Datasets:We used Oxford-IIIT pet datasets. The datasets consists of 37 category pet dataset consisting of 200 images for each class.

Architecture:We used the famous U-net architecture  for our task. The U-net is U shaped architecture and has two major parts:contracting part(downsampling) having general convolution process and expansive part(upsampling) with transposed 2D convolution layers. We inserted two PT layers with single transformation matrix: One PT layer after the first double convolution of contracting part and the next PT layer after the last double convolution of expansive part.The size of input images were 128*128*3.

Experiment Scheme:We trained two models: the baseline U-net model and the model with PT layers under the same training configuration. We trained the model for 200 epochs and evaluated both models using dice coefficient.

Results:The average dice coefficient for two models were found to be 0.75 and 0.76 respectively. The model with PT layer has better performance and better visualization results as shown in Figure below.

<img width="566" alt="seg" src="https://user-images.githubusercontent.com/16822926/203696367-42452bc9-e622-4e8c-af87-5fcbdbedd0c4.png">

Figure : Visualizations of segmentation results. First column: input images; Second column: target output; Third column: outputs of baseline U-net; Last column: outputs of the model with PT layer.



## Paper Citation
Nishan Khatri, Agnibh Dasgupta, Yucong Shen, Xin Zhong,Frank Y.Shih. 2022. Perspective Transformation Layer. 2022 International Conference on Computational Science & Computational Intelligence (CSCI'22).

https://arxiv.org/abs/2201.05706





