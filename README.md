# ResNet-Style CIFAR-10 Image Classifier (PyTorch)

A ResNet-style convolutional neural network built in PyTorch and trained on the CIFAR-10 image dataset.

## Highlights
- Custom CNN with residual blocks, downsampling bridge blocks, and batch normalization
- Reached about 90% test accuracy using data augmentation, input normalization, and a learning-rate schedule
- Complete training and evaluation pipeline with timing, loss and error tracking, and softmax predictions
- Visualizes a random test image alongside the model's confidence scores for each class

## Results
| Metric | Value |
| --- | --- |
| Test accuracy | ~90.15% |

See `train.ipynb` for full training logs and evaluation.

## Model
- Three stages of 7 residual blocks each (16, 32, and 64 channels), connected by bridge blocks that halve the image size and double the channels
- Average pooling, then a linear classifier head (10 classes)
- About 750,000 parameters

## Training Setup
- 50,000 training images and 10,000 test images
- Horizontal-flip data augmentation and input normalization
- SGD with momentum 0.9 and weight decay 5e-4, batch size 150
- Learning rate 0.05, dropped to 0.005 at epoch 50, for 80 epochs total
- Trained on a GPU in Google Colab

## Files
- `model.py`: network definition (residual blocks, bridge blocks, classifier head)
- `train.ipynb`: end-to-end training and evaluation notebook
- `utils.py`: helpers for the error metric and plotting, provided as part of the course
- `requirements.txt`: Python dependencies

## Data
The preprocessed CIFAR-10 tensors were provided for the course and are not included in this repo. The notebook expects these files:

```
data/cifar/train_data.pt
data/cifar/train_label.pt
data/cifar/test_data.pt
data/cifar/test_label.pt
```

## How to Run
1. Place the data files in `data/cifar/` as shown above.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `train.ipynb` in Google Colab or Jupyter (a GPU is recommended) and run the cells top to bottom.

## Skills Demonstrated
Python, PyTorch, CNNs, residual networks, data augmentation, GPU training, model evaluation, visualization
