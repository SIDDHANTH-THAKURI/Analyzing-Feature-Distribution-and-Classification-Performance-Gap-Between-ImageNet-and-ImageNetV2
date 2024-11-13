# 🎨 Analyzing Performance Gaps in Image Classification Models

Welcome to the **Analyzing Performance Gaps in Image Classification Models** repository! This project dives into the fascinating area of model generalization across datasets, comparing **ImageNet** and **ImageNetV2** performance using deep features extracted from the EVA-02 model.

## 🚀 Project Highlights

- **Explore** feature distributions and analyze performance differences between ImageNet and ImageNetV2.
- **Visualize** distribution shifts in the feature space using dimensionality reduction techniques like **t-SNE** and **UMAP**.
- **Train and Evaluate** models with techniques like incremental learning to handle large, high-dimensional data effectively.

## 🌟 Table of Contents

- [Project Highlights](#-project-highlights)
- [Motivation](#-motivation)
- [Dataset Details](#-dataset-details)
- [Key Features](#-key-features)
- [Getting Started](#%EF%B8%8F-getting-started)
- [Data Requirements](#-data-requirements)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Directions](#-future-directions)
- [Acknowledgments](#-acknowledgments)

## 💡 Motivation

In real-world applications—such as autonomous vehicles and medical imaging—models need to generalize well across diverse datasets. This project addresses the **performance gap** between models trained on ImageNet and tested on ImageNetV2. Through deep analysis and visualization, we aim to:

- Identify the **factors contributing to performance degradation**.
- Develop strategies to **enhance model robustness** across varied datasets.

## 📂 Dataset Details

The project utilizes three main datasets:

- **Training Set**: `train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` – 1.28M samples
- **Test Set 1 (ImageNet)**: `val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` – 50,000 samples
- **Test Set 2 (ImageNetV2)**: `v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` – 10,000 samples

Each dataset contains **high-dimensional features**, extracted from the EVA-02 model, along with class labels and image paths.

## 🔑 Key Features

- **Incremental Learning**: Handle large data by loading it in chunks, using models like `SGDClassifier`.
- **Model Evaluation**: Compare model accuracy and generalization across ImageNet and ImageNetV2.
- **Feature Space Visualization**: Use dimensionality reduction (`t-SNE`, `UMAP`) for in-depth visualization of distribution shifts.
- **Hyperparameter Tuning**: Grid Search for optimizing model performance (limited due to computational constraints).

## 🛠️ Getting Started

### Prerequisites

Make sure you have **Python 3.7+** installed. Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn
Running the Code
Place your dataset files in the same directory as the scripts, adjust file paths in each script if necessary, and run each script from the command line:

bash
Copy code
python sgd_classifier_model.py
Hardware Requirements
For smooth execution, it’s recommended to use a system with 16GB+ RAM. Alternatively, leverage cloud resources like Google Colab Pro with GPU support.
```
## 📂 Data Requirements

To run this project and replicate the results, you will need access to specific datasets and a pretrained model for feature extraction. Follow the instructions below to gather the necessary resources:

### 1. ImageNet Dataset

**Description**: The original ImageNet dataset, which includes a large collection of labeled images used widely for training and evaluating computer vision models.

**Usage**: This dataset is needed to create the **Training Set** and **Test Set 1** used in this project.

**Access**: ImageNet requires a registration and application process due to licensing restrictions. You can request access to the dataset through the official [ImageNet website](https://www.image-net.org/).

**File Requirements**:

- `train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (Training Set - 1.28M samples)
- `val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (Validation Set - 50,000 samples)

### 2. ImageNetV2 Dataset

**Description**: ImageNetV2 is a dataset created to test model generalization by using images that resemble ImageNet’s but are collected separately. This project uses it as **Test Set 2**.

**Access**: You can download ImageNetV2 from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imagenet_v2) or its [GitHub repository](https://github.com/modestyachts/ImageNetV2).

**File Requirement**:

- `v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (ImageNetV2 Test Set - 10,000 samples)

### 3. EVA-02 Model for Feature Extraction

**Description**: The EVA-02 model is used in this project to extract high-dimensional features from the images in both ImageNet and ImageNetV2 datasets.

**Access**: Check platforms like [Hugging Face Model Hub](https://huggingface.co/) or refer to the original EVA-02 paper or repository to access the pretrained model.

## 📁 Project Structure

. ├── README.md # Project description and setup instructions ├── sgd_classifier_model.py # SGD classifier model training and evaluation ├── logistic_regression_model.py # Logistic Regression model script ├── random_forest_model.py # Random Forest model script (sample data only due to size constraints) ├── visualizations.py # Generates feature space visualizations ├── requirements.txt # Dependencies for the project └── data/ # Directory for dataset CSV files

markdown
Copy code

## 📊 Results

### Visualization Samples

- **Feature Distribution Shifts**: Significant shifts in some features between Test Set 1 (ImageNet) and Test Set 2 (ImageNetV2).
- **t-SNE and UMAP Clusters**: Visualized clusters revealing unique feature patterns in each test set, impacting model performance.

### Accuracy Comparison

- **SGDClassifier**: 51% (Test Set 1) vs. 44% (Test Set 2)
- **Logistic Regression**: 48% (Test Set 1) vs. 42% (Test Set 2)

## 🔭 Future Directions

- **Hyperparameter Tuning**: Deeper tuning to refine model accuracy across datasets.
- **Distribution Shift Mitigation**: Explore domain adaptation techniques to handle shifts.
- **Class Imbalance Solutions**: Apply resampling or class-weighting methods for balanced representation.
- **Alternative Models**: Test with models optimized for high-dimensional data.

## 🙏 Acknowledgments

Special thanks to the **School of Computing and Information Technology**, University of Wollongong, for resources and guidance, and to the contributors for their dedication to advancing big data analytics and machine learning.

---

<p align="center" style="font-size: 1.2em;">Happy coding! ✨</p>
