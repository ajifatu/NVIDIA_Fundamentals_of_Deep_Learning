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

- To solve problem in last notebook, we create a more sophisticated CNN model (it understands a greater variety of model layers for better performance)
- We started with reshaping our dataset to allow our convolutions to associate groups of pixels and detect important features.
- Our convolutional model: Input -> **Convolution** -> Max Pooling -> **Convolution** -> Dropout -> Max Pooling -> **Convolution** -> Max Pooling -> Dense -> Dense -> Output
  - 3 convolutions layers, each applying a convolution operation to extract features from the input images, followed by batch normalization, ReLU activation, and pooling to reduce spatial dimensions while retaining important features.
  - The output of the last convolutional layer is flattened and passed through two fully connected layers (Dense).
- Model is significantly improved! The training accuracy is very high, and validation accuracy improved as well.
- _But_, validation accuracy jumping around. There's more work to be done!
