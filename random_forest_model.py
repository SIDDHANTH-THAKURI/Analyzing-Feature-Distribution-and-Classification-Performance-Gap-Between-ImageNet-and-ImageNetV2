# random_forest_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import gc

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path, nrows=None):
    """
    Load data from a CSV file.
    """
    data = pd.read_csv(file_path, nrows=nrows)
    return data

def prepare_data(data):
    """
    Prepare data by separating features and labels.
    """
    # Exclude 'path' and 'label' columns
    features = data.drop(['path', 'label'], axis=1)
    # Convert features to float32
    features = features.astype(np.float32)
    # Convert labels to int32
    labels = data['label'].astype(np.int32)
    return features, labels

def sample_data(train_file, sample_size=50000):
    """
    Sample a subset of the training data.
    """
    print(f"Sampling {sample_size} instances from the training data...")
    data = pd.read_csv(train_file, nrows=sample_size)
    return data

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier.
    """
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_on_test_set(test_file, model, scaler, dataset_name):
    """
    Evaluate the model on a test set.
    """
    print(f"Evaluating on {dataset_name}...")
    data = load_data(test_file)
    X_test, y_test = prepare_data(data)
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return y_test, y_pred

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """
    Plot confusion matrix for the given predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', annot=False)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()

def main():
    # File paths
    train_file = 'train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    test1_file = 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    test2_file = 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'

    # Sample training data
    sample_size = 50000  # Adjust based on your memory capacity
    train_data = sample_data(train_file, sample_size=sample_size)
    X_train, y_train = prepare_data(train_data)
    del train_data
    gc.collect()

    # Initialize scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    del X_train
    gc.collect()

    # Train model
    model = train_random_forest(X_train_scaled, y_train)
    del X_train_scaled, y_train
    gc.collect()

    # Evaluate on Test Sets
    y_true_test1, y_pred_test1 = evaluate_on_test_set(test1_file, model, scaler, "Test Set 1")
    y_true_test2, y_pred_test2 = evaluate_on_test_set(test2_file, model, scaler, "Test Set 2")

    # Plot confusion matrices
    plot_confusion_matrix(y_true_test1, y_pred_test1, 'Test_Set_1')
    plot_confusion_matrix(y_true_test2, y_pred_test2, 'Test_Set_2')

    # Feature Importance Visualization
    if hasattr(model, 'feature_importances_'):
        print("Plotting feature importances...")
        # Load feature names from training data
        feature_cols = [col for col in X_train.columns]
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        # Plot top 30 features
        plt.figure(figsize=(15, 8))
        sns.barplot(data=feature_importance.head(30), x='importance', y='feature')
        plt.title('Top 30 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

if __name__ == "__main__":
    main()
