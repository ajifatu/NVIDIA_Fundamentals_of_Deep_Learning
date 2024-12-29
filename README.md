# NVIDIA_Fundamentals_of_Deep_Learning

## What we did:

### 1st notebook: Image Classification with the MNIST Dataset

- The Deep Learning "Hello world" project!
- Built and trained a simple neural network for image classification to correctly classify hand-written digits.
- Tools: Pytorch, TorchVision, Matplotlib

### 2nd notebook: Image Classification of an American Sign Language Dataset

- Did quite same as above, but this time we got dataset from Kaggle, loaded, normalized and customized it to pass it to a dataLoader.
- Encountered an issue with the accurac: Training accuracy got to a fairly high level while validation accuracy was not as high (even after 20 epochs). Thing is model is memorizing the dataset, but not gaining a robust and general understanding of the problem. That's called **overfitting**!  
  Solution in next notebook ?
