# Object-Detection with RCNN

---

This repository contains a Tensorflow 2 implementation of a Regional Convolutional Neural Network (RCNN), trained to detect
aeroplanes from aerial images. The RESISC45 dataset's aeroplane images are used for this task, with the annotations
taken from this [GitHub repo](https://github.com/1297rohit/RCNN/blob/master/Airplanes_Annotations.zip).

We use VGG16 as the backbone, and replace the final layer (of 1000 classes) with a 2 neuron Dense layer
for the output. Training approximately converges after 100-120 epochs, with the best model's weights
restored after early stopping. The predictions for some images in the test set are shown below.

![One](./output_images/pred1.png)
![Two](./output_images/pred2.png)
![Three](./output_images/pred3.png)
![Four](./output_images/pred4.png)
![Five](./output_images/pred5.png)