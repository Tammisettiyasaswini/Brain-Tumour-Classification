# Brain-Tumour-Classification
This project classifies brain MRI scans into Brain Tumor or Healthy using deep learning. We implemented a custom CNN and compared it with InceptionV3, ResNet50, and MobileNetV2. Images were preprocessed and trained with transfer learning. ResNet gave the best accuracy, MobileNet was fastest, proving DL’s power in medical imaging.
# Brain Tumor Classification 🧠  

This project builds a deep learning model to classify brain tumors (glioma, meningioma, pituitary) from MRI scans using CNNs and transfer learning (InceptionV3/EfficientNet).  

## 📂 Dataset  
We used the *Brain Tumor MRI Dataset* available on Kaggle:  
[👉 Dataset Link](/kaggle/input/brain-tumor-classification-mri) 
[👉 Dataset Link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) 
[👉 Dataset Link](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data)  
## Keras (/kaggle/working/brain_tumor_model.keras)

## ⚙ Steps  
1. Data Acquisition  
2. Preprocessing (resize, normalize, augmentation)  
3. Model (Transfer Learning with CNNs)  
4. Training & Fine-tuning  
5. Evaluation (accuracy, confusion matrix, precision, recall, F1)  

## 📊 Results  
-Final Train Accuracy: 99.57%
 Final Validation Accuracy: 98.21% 

## 🔗 References  
- [TensorFlow](https://www.tensorflow.org/)  
- [Keras](https://keras.io/)  
