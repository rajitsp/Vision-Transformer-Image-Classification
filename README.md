# Vision-Transformer-Image-Classification

This project explores the use of Vision Transformers (ViT) for image classification tasks, specifically on the EuroSAT dataset. The goal is to build and train a ViT model and compare its performance with traditional Convolutional Neural Network (CNN) models.

## Overview

Transformers, originally introduced for Natural Language Processing (NLP) tasks, have recently been adapted for computer vision applications. Vision Transformers treat images as sequences of patches and process them using the transformer architecture. This approach has shown promising results and has the potential to outperform traditional CNNs in image classification tasks.

In this project, we implement a Vision Transformer model and compare its performance with a baseline CNN model and the popular ResNet50 architecture. The results are evaluated on the EuroSAT dataset, which consists of 27,000 labeled 64x64 satellite images classified into 10 different land use classes.

## Dataset

The [EuroSAT](https://github.com/phelber/EuroSAT) dataset is used for training and evaluating the models. It contains 27,000 labeled satellite images of size 64x64 pixels, categorized into 10 different land use classes such as highways, forests, rivers, and various vegetation types.

## Approach

1. **Data Preprocessing**: The images are converted to pixel values, and data augmentation techniques are applied.
2. **Patch Embedding**: The images are divided into patches, and each patch is converted into a 1D sequence.
3. **Positional Encoding**: Positional embeddings are added to the patch sequences to preserve the spatial information.
4. **Vision Transformer**: The patch sequences are processed by the Vision Transformer model, which consists of a series of transformer encoder layers.
5. **Classification Head**: The output of the transformer is passed through a classification head to obtain the final class predictions.

## Models

1. **Vision Transformer (ViT)**: The main model implemented in this project, based on the Vision Transformer architecture.
2. **CNN Baseline**: A 5-layer Convolutional Neural Network (CNN) model with 3x3 kernels, ReLU activation, max pooling, and dropout regularization.
3. **ResNet50**: The popular ResNet50 architecture is used as a benchmark for comparison.

## Results

The models were trained and evaluated on the EuroSAT dataset. The following table summarizes the key results:

| Model                 | Trainable Parameters | Time per Epoch | Validation Accuracy | Validation Loss |
|---------------------- |----------------------|-----------------|----------------------|-----------------|
| CNN (25 epochs)       | 157,018              | 8-9s            | 87.20%              | 0.4658          |
| ResNet50 (25 epochs)  | 17,132,490           | 22-23s          | 94.69%              | 0.2988          |
| ViT (3 epochs)        | 85,806,346           | >=5000s (83min) | 98.46%              | 0.0728          |

The Vision Transformer model achieved the highest validation accuracy of 98.46%, outperforming both the baseline CNN and the ResNet50 model. However, it required significantly more computational resources and training time compared to the other models.

## Lessons Learned

- Vision Transformers can achieve state-of-the-art performance in image classification tasks, but they require large amounts of data and computational resources for training.
- The fundamental approach of treating images as sequences of patches and processing them with a transformer architecture showed promising results.
- While CNNs are still widely used, there is a shift towards transformer-based architectures in computer vision due to their potential for better performance.

## Future Work

The field of computer vision is rapidly evolving, and transformers are gaining traction as an alternative to traditional CNN architectures. Some potential future directions include:

- Exploring hybrid models that combine transformers and CNNs to leverage the strengths of both architectures.
- Investigating techniques to reduce the computational cost and memory requirements of Vision Transformers.
- Applying Vision Transformers to other computer vision tasks such as object detection, segmentation, and tracking.
- Utilizing pre-trained Vision Transformer models and transfer learning to improve performance and reduce training time.


## Acknowledgments

- The EuroSAT dataset was introduced in the paper "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" by Helber et al.
- The Vision Transformer architecture was introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.
