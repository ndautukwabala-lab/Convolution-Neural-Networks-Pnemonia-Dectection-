# Convolution-Neural-Networks-Pnemonia-Dectection-
A CNN-based pneumonia detection system trained on chest X-ray images using transfer learning. Covers data preprocessing, model training, fine-tuning, and performance evaluation with visual results such as training curves and a confusion matrix.

# Pneumonia Detection Using Convolutional Neural Networks (DenseNet121)

A deep learning project that applies a convolutional neural network to classify chest X-ray images as Pneumonia or Normal using transfer learning with DenseNet121. The work includes preprocessing, exploratory analysis, model training, fine-tuning, and evaluation through visual performance metrics.

## Dataset
Kaggle: Chest X-Ray Images (Pneumonia)  
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Images are resized to 224×224, normalized, augmented, and batched for training, validation, and testing.

## Project Structure
├── CNN_Pneumonia.ipynb
├── sample_xrays.png
├── confusion_matrix.png
├── train_accuracy.png
├── train_loss.png
└── README.md


## Model Summary
- DenseNet121 backbone (ImageNet pre-trained)
- Transfer learning + fine-tuning
- Adam optimizer, binary cross-entropy loss
- Data augmentation applied during training

## How to Run
1. Download dataset via KaggleHub:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

2. Open the notebook in Google Colab.

Run preprocessing, training, and evaluation cells.

Results (Test Set)
Metric	Score
Accuracy	0.7356
Precision	0.7034
Recall	0.9974
AUC	0.7551
Visual Outputs
Sample X-rays

Confusion Matrix

Training Accuracy

Training Loss

Future Work

Add Grad-CAM visual explanations

Balance dataset or apply focal loss

Test additional architectures (EfficientNet, ResNet50)

Deploy model via Streamlit

License

For academic and research use.
