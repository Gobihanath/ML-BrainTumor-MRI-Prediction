# ML-BrainTumor-MRI-Prediction

## Brain Tumor Classification Using Convolutional Neural Networks
### Introduction

The brain tumor classification model implemented in this project is based on a Convolutional Neural Network (CNN) designed to classify MRI images of brain tumors. The model aims to assist in the early detection and classification of brain tumors, which is crucial for timely and effective treatment. The dataset used includes various types (44 types) of brain tumor images, which were divided into training, validation, and test sets to ensure a robust evaluation of the model.

### Dataset
The dataset used in this project is composed of MRI images of brain tumors, categorized into different classes. The images are divided into training, validation, and test sets to facilitate model training and evaluation. The dataset is stored in the following directory structure: https://drive.google.com/drive/folders/1MSgnk1u9amSvoSFHsVDASJwYOdxfdOvd 

kaggle dataset link : https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c 

### Key features
Key features of the model include data augmentation to improve generalization, dropout layers to prevent overfitting, and batch normalization to stabilize and accelerate the training process. The model was trained using a categorical cross-entropy loss function and the Adam optimizer, which are standard choices for multi-class classification problems.

The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results demonstrate high accuracy in classifying brain tumors, indicating the model's effectiveness. Additionally, confusion matrices for both validation and test datasets provide detailed insights into the model's performance across different classes.






### Future work may include:

- Expanding the dataset to include more diverse images.
- Further tuning the model hyperparameters for better performance.
- Implementing advanced techniques like transfer learning.
- Developing a user-friendly interface for easier deployment in clinical settings.
