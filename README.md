## Image Classification using CIFAR10 Dataset with TensorFlow/Keras
This project aims to build an image classification model using the CIFAR10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to correctly classify these images into their respective categories using machine learning algorithms and deep learning techniques.

### Dataset Description
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset:

`airplane`										

`automobile`										

`bird`										

`cat`										

`deer`										

`dog`										

`frog`										

`horse`										

`ship`									

`truck`										

### Model Architecture
The model was built using TensorFlow and Keras, with a convolutional neural network (CNN) architecture. The CNN consists of multiple convolutional layers, pooling layers, and fully connected layers, with a final softmax layer to output the class probabilities. The model was trained using the Adam optimizer and categorical cross-entropy loss function.

### Evaluation Metrics
The model was evaluated using several metrics, including accuracy, precision, recall, and F1-score. The accuracy metric measures the overall percentage of correct predictions, while the precision metric measures the percentage of true positives out of all positive predictions. The recall metric measures the percentage of true positives out of all actual positive cases, while the F1-score is a weighted average of the precision and recall metrics.

### Conclusion
In this project, we have built an image classification model using the CIFAR10 dataset and TensorFlow/Keras. The model achieved high accuracy and performed well on various evaluation metrics. This model can be further improved by fine-tuning the hyperparameters or by using transfer learning techniques.
