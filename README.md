# 18-796 HW3 Semantic Segmentation

In this assignment, we'll be building our own DeepLab network, a framework designed for high resolution, precise image segmentation, and using it to predict a categorical label for every single pixel in an image.

This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. 

To train a segmentation network, you will need an annotated dataset where a training pair contains an RGB image and the annotated segmentation map. Each pixel is labeled by a categorical number similar to the classification.

* To finish this assignment, you need to submit a zip file containing both the finished code and a report.

## Your tasks
* Build a data processing pipeline and visualize the annotation
* Build your DeepLabV3 network
* Train and evaluate your own network on PascalVOC dataset
* Apply the latest segment-anything-model.

### 0. environments and packages
We recommend you start with an Anaconda environment but feel free to use anything. Then in your environment, run 

```bash
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
Your system should be equipped with an Nvidia GPU, and plz follow the official PyTorch instruction to install the GPU version properly. We have tested the code with Pytorch 1.10 & TorchVision 0.11.0 for this assignment.

## 1. Prepare Data Pipeline (20pts)
In the ``datasets/`` folder, create a subdirection ``data``. Download and unzip the PascalVOC dataset. We will be using the [2012 branch](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). Your path should be:

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```
1. (5pts) In ``datasets/voc.py``, complete the ``VOCSegmentation`` class. Specifically, please finish the ``_getitem_`` method and ``decode_target`` method. 
2. (5pts) How many categories are there in the dataset? In the training set, do you think this is a class-balanced set? And what can be the potential challenges while training a model on this dataset? 
3. (5pts) Semantic Segmentation utilizes Accuracy and Mean IoU as evaluation metrics. Can you give an example to show why mIoU is a better metric than accuracy? 
4. (5pts) Write a file `visualize.py` that reads images and annotation/prediction and plot the ground truth/prediction. Pick any 20 training images that covers all available categories. Plot the ground truth annotations side-by-side with the training images and show them in your report. (You can organize this as a large picture with 10 columns x 4 rows) 

## 2. Build Segmentation Network (30pts) 
Please read the original [DeepLabV3](https://arxiv.org/pdf/1706.05587.pdf) & [DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf) paper and implement the network.
1. (15pts) In ``network/_deeplab.py``, complete ``ASPPConv``, ``ASPPPooling`` and ``ASPP``. 
2. (15pts) Complete ``DeepLabHead`` & ``DeepLabHeadPlus``. What are their differences? Please write your understanding in the report.


## 3. Training and Evaluation (35pts)
In ``main.py`` file, please complete the training loop and train the networks. 

1. (5pts) Build the SGD optimizer with a default `learning_rate=0.01` and `momentum=0.9`. Since we will be using an ImageNet pretrained ResNet, we want to scale down the learning rate on the __backbone__ component. Set up an optimizer such that the learning rate of the __backbone__ is 0.1x the main learning rate. 

2. (5pts) Build the learning rate schedular. We will be using a step learning rate schedular that reduce the learning rate by 0.9x every 1,000 iterations. 

3. (10pts) Train the DeepLabV3+ with resnet50 backbone (imagenet pretrained) on PascalVOC train split for 5k iteration, with CrossEntropyLoss. Plot the evaluation mIOU with an interval of 1 epoch in your report. (You should be able to train the model with a batch size of 8. In our test run, GPU memory usage is 6861MB.)

4. (10pts) Based on what you have learned in the dataset, train the same network using the same settings, but in this time, with a different loss function. Write your loss function in `utils/loss.py`. What loss function do you propose to use and why?  Plot the evaluation mIOU with an interval of 1 epoch in your report. (Hint: What do you learn from the Detection assignment when it comes to class-imbalanced training?)

5. (5pts) For both models, report the best performances and pick 5 images in problem 1, show the ground-truth vs predictions side-by-side. There should be 4 columns, | RGB Image | Ground Truth annotation | Model 1 prediction | Model 2 prediction |


## 4. Segment-Anything-Model (15pts)
[Segment-Anything-Model](https://arxiv.org/abs/2304.02643) is recently proposed by Meta AI Research that produces complete high-quality object masks. Please follow the [installation](https://github.com/facebookresearch/segment-anything) instruction, and download the pretrained model checkpoint. You can also modify the provided [Google Colab script](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb) as well to save some AWS credits.

1. Run the model on the same 5 images. What do you think are the differences between sementic segmentation and SAM?


## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[3] [Segment Anything](https://arxiv.org/abs/2304.02643)