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

### 4th notebook A: Data augmentation on the ASL model

In order to teach our model to be more robust when looking at new data, we're going to increase the size and variance in our dataset: **_data augmentation_**

- Did the same data processing as for the previous model
- Since our CNN model uses a sequence of repeated layers, we took advantage of that pattern to make our own (convolution block ) module that we used as a layer in our sequential model
- Set up data augmentation with TorchVision's Transforms module before the training.
  Transformations made: RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter
  Made sure to visualize each transformation made with plt for a better understanding
- One thing changed on our train function: Before passing our images to our model, we applied our random_transforms which is a sequence of those random transformations with Compose
- Result: validation accuracy is higher, and more consistent. This means that our model is no longer overfitting in the way it was; it generalizes better, making better predictions on new data. (Finally ? Mdrr)
- Last moves before moving to next notebook:
  - Moved our `_get_batch_accuracy_` function and `_MyConvBlock_` custom module to utils
  - Saved our trained model to disk

### 4th notebook B: Deploying our Model

Model well trained, let's use it and make predictions on new images. That's called **_Inference_** !

- First, we loaded the model since we saved it later for convenience making sure to first import our MyConvBlock
- To match the shape of the data the model was trained on, we scaled images. They are [1, 184, 186] shape while our training images were [1, 28, 28].
  We used TorchVision's Transforms module again
  Converted image to float with ToDtype while setting scale to True in order to convert from [0, 255] to [0, 1]
  Resized the image to be 28 x 28 pixels
  Converted the images to Grayscale (No effect since our images are loaded grayscale, but good to know)
- Now, predictions:
  - First thing first, we printed image with matplotlib
  - Loaded it in grayscale mode by using TorchVision's read_image function ImageReadMode and setting ImageReadMode to GRAY
  - Transformed it to scale to model trained images (point below)
  - Batched it since our model expects a batch of images using `_.unsqueeze(0)_`
  - Sent image to correct device (the GPU)to make sure input tensor is on the same device as the model
  - **Predictions**: output = model(image), image being final output of the steps below
  - Found max index: Predictions are in the format of a 24 length array. The larger the value, the more likely the input image belongs to the corresponding class. So we founded index of the max.  
    `prediction = output.argmax(dim=1).item()`
  - Converted prediction to letter: Each element of the prediction array represents a possible letter in the sign language alphabet. (j and z are not options because they involve moving the hand, and we're only dealing with still photos)  
    `alphabet = "abcdefghiklmnopqrstuvwxy"`  
    `predicted_letter = alphabet[prediction]`
