## Introduction
This project is an implementation of a neural network and training the same on the CIFAR-10 dataset, which is an emerging standard in computer vision. The architecture to be used for this neural network shall be that based on VGG16. The big goal, however, remains that of understanding neural network compression techniques, since doing so will lead to model performance optimization through the reduction of parameters, thereby increasing speed and efficiency in both phases of training and inference.

## Data Preparation
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 diverse classes. Each class has 6,000 images. This data is further divided into a training set of 50,000 and test images of 10,000. To avoid overfitting and enhance the generalization capability of the model, a data augmentation strategy will be applied before training, which includes image flipping and cropping. Moreover, the validation set Make up 10% of the training data to allow for more robust model evaluation during training.

## Hyperparameters and Training
The neural network is trained using a predefined set of hyperparameters. Initially, the learning rate is set at 0.001 and later increased to 0.01 to enhance the model's learning capabilities. The training process involves iterating over the dataset multiple times (epochs) until the model achieves an accuracy of approximately 90% on the evaluation data. The architecture of the VGG16 model is instantiated, and the training process is carefully monitored to adjust the learning rate and other parameters as needed. Data augmentation techniques are applied during training to further improve the model's performance and robustness.

![image](https://github.com/fmirzadeh99/Implementation-and-compression-of-the-VGG16-model/assets/169579231/8fa85c65-7527-498f-a8fb-7931d16afa8c)
|:-:|
|Loss and Accuracy graph of the VGG16 model on the Dataset

## Evaluation Metrics and Results
Upon reaching the desired accuracy, the model's performance is thoroughly evaluated using error and accuracy metrics. Visualization plots are created to illustrate the model's progress and highlight areas where high accuracy has been achieved. Computing singular information center values by all network layers is done for both the training and test datasets. Graphs of their changes across different datasets can also be provided. Such a comprehensive evaluation process could help in understanding the model behavior and in the effectiveness of compression and optimization of neural network architectures.

![image](https://github.com/user-attachments/assets/45c7c6de-835c-42be-a14c-bdf20feb4c27)
|:-:|
|Test Center Separation Index (CSI) Across Layers
