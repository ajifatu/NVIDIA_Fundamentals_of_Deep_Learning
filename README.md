# NVIDIA_Fundamentals_of_Deep_Learning

## What we did:

### 1st notebook: Image Classification with the MNIST Dataset

- The Deep Learning "Hello world" project!
- Built and trained a simple neural network for image classification to correctly classify hand-written digits.
- Tools: Pytorch, TorchVision, Matplotlib

### 2nd notebook: Image Classification of an American Sign Language Dataset (ASL)

- Did quite same as above, but this time we got dataset from Kaggle, loaded, normalized and customized it to pass it to a dataLoader.
- Encountered an issue with the accuracy: Training accuracy got to a fairly high level while validation accuracy was not as high (even after 20 epochs). Thing is model is memorizing the dataset, but not gaining a robust and general understanding of the problem. That's called **overfitting**!  
  Solution in next notebook ?

### 3rd notebook: ASL model with Convolutional Neural Networks

To solve problem in last notebook, we create a more sophisticated CNN model (it understands a greater variety of model layers for better performance)

- We started with reshaping our dataset to allow our convolutions to associate groups of pixels and detect important features.
- Our convolutional model: Input -> **Convolution** -> Max Pooling -> **Convolution** -> Dropout -> Max Pooling -> **Convolution** -> Max Pooling -> Dense -> Dense -> Output
  - 3 convolutions layers, each applying a convolution operation to extract features from the input images, followed by batch normalization, ReLU activation, and pooling to reduce spatial dimensions while retaining important features.
  - The output of the last convolutional layer is flattened and passed through two fully connected layers (Dense).
- Model is significantly improved! The training accuracy is very high, and validation accuracy improved as well.
- _But_, validation accuracy jumping around. There's more work to be done!

### 4th notebook: Data augmentation on the ASL model

In order to teach our model to be more robust when looking at new data, we're going to increase the size and variance in our dataset: **_data augmentation_**

- Did the same data processing as for the previous model
- Since our CNN model uses a sequence of repeated layers, we took advantage of that pattern to make our own (convolution block ) module that we used as a layer in our sequential model
- Set up data augmentation with TorchVision's Transforms module before the training.
  Transformations made: RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter
  Made sure to visualize each transformation made with plt for a better understanding
- One thing changed on our train function: Before passing our images to our model, we applied our random_transforms which is a sequence of those random transformations with Compose
- Result: validation accuracy is higher, and more consistent. This means that our model is no longer overfitting in the way it was; it generalizes better, making better predictions on new data. (Finally ? Mdrr)
- Last moves before moving to next notebook:
  - Moved our _get_batch_accuracy_ function and _MyConvBlock_ custom module to utils
  - Saved our trained model to disk
