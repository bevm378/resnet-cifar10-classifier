# ResNet-Style CIFAR-10 Image Classifier (PyTorch)

A simplified ResNet-style convolutional neural network trained on the CIFAR-10 dataset using PyTorch and GPU acceleration.

## Highlights
- Custom CNN with residual blocks and batch normalization  
- Trained on CIFAR-10 to ~90% test accuracy using tuning, batching, and learning-rate optimization  
- Complete training/evaluation pipeline with timing, loss/error tracking, and softmax-based predictions  
- Random test-image visualization with class confidence scores  

## Results

Final test accuracy: ~90.15%
(See `train.ipynb` for full logs and evaluation)

## Files
- `model.py` – network definition (residual blocks + classifier)  
- `train.ipynb` – end-to-end training and evaluation notebook  
- `utils.py` – helper functions for loading data and plotting  

## Skills Demonstrated
Python, PyTorch, CNNs, Deep Learning, GPU training, Model evaluation, Visualization
