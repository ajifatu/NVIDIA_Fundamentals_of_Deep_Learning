# NVIDIA_Fundamentals_of_Deep_Learning

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

### 5th notebook A: Pretrained models

In this notebook, we created "An Automated Doggy Door". Since we need a lot of data to train a DL model, we used the `VGG16` network _pre-trained_ on the ImageNet dataset (a massive dataset, including lots of animals).

- We first loaded the model, and set weights to `DEFAULT`
- Images should same dimensions as ones model was trained with. Thankfully, the pretrained weights come with their own transforms. So we applied those default transformations and turn it into a batch (required input)
- We had 1000 possible categories that the image would be placed in (the output shape is 1000). Which is a lot, we just needed the dogs and cats categories. So, we used the `torch.topk` function to give us the top 3 predictions and and checked if argmax index corresponded to dogs or cats

### 5th notebook A: Transfer learning / Finetuning

In last notebook, we used a pre-trained model with no training necessary since it was doing exactly what we needed: Check the type of animal and we on that base made a treatment. But what if we don't find one and neither don't have enough data to train one ? **Tranfer Learning**: take a pre-trained model and retrain it on a task that has some overlap with the original training task. We used the same model as previously, the VGG16 model; goal being to make an automatic doggy door for a dog named Bo, the United States First Dog between 2009 and 2017. (mdrrr?) We have only 30 pictures of Bo

- We started with downloading the pre-trained model and setting weights to DEFAULT
- We hav the 1000 possible classes in the dataset. In our case, we just wanted to make a different classification: is this Bo or not? We then added new layers to specifically recognize Bo. For that:
  - We freezed the model's pre-trained layers. `vgg_model.requires_grad_(False)`
  - Added new layers: We just added a Linear layer connecting all 1000 of VGG16's outputs to 1 neuron `nn.Linear(1000,1)`
  - Compiled the model with loss and metrics options. Since we have a binary classification problem (Bo or not Bo), and we used binary crossentropy `BCEWithLogitsLoss`
- We did some data augmentation using preprocessing transforms from the VGG weights to match with model parameters, customed dataset and created dataLoaders
- We trained model making sure only our newly added layers were learning by checking last set of gradients
- Result: Both training and validation accuracy were quite high, even with a tiny dataset (30 images): Power of transfer learning. THough, we explored another method: Fine-tuning
- To do so, we unfreezed the entire model, and trained it again with a very small learning rate.  
   `vgg_model.requires_grad_(True) `  
  `optimizer = Adam(my_model.parameters(), lr=.000001) `
