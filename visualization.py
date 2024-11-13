# visualizations.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import gc
import umap
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class ImageNetVisualizer:
    def __init__(self, chunk_size=5000):
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self.color_palette = sns.color_palette("Set2")

        # File paths for the provided CSV files
        self.train_file = 'train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
        self.test1_file = 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
        self.test2_file = 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'

    def setup_plot_style(self):
        """Set up consistent plot styling"""
        plt.style.use('ggplot')
        sns.set_palette(self.color_palette)
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12

    def load_data_chunk(self, file_path, nrows=None):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(file_path, nrows=nrows)
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            return None

    def plot_feature_distributions_comparison(self):
        """Compare feature distributions between test sets"""
        print("Plotting feature distributions comparison...")
        self.setup_plot_style()

        # Load sample chunks from both test sets
        test1_data = self.load_data_chunk(self.test1_file, nrows=5000)
        test2_data = self.load_data_chunk(self.test2_file, nrows=5000)

        if test1_data is None or test2_data is None:
            return

        # Extract features (excluding path and label columns)
        feature_cols = [col for col in test1_data.columns if col not in ['path', 'label']]
        test1_features = test1_data[feature_cols]
        test2_features = test2_data[feature_cols]

        # Select random 6 features for visualization
        selected_features = np.random.choice(feature_cols, size=6, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        for i, feature in enumerate(selected_features):
            sns.kdeplot(data=test1_features[feature], ax=axes[i], label='Test Set 1', color='#1f77b4', shade=True)
            sns.kdeplot(data=test2_features[feature], ax=axes[i], label='Test Set 2', color='#ff7f0e', shade=True)
            axes[i].set_title(f'Feature {feature} Distribution')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig('feature_distributions_comparison.png')
        plt.close()

        # Clear memory
        del test1_data, test2_data, test1_features, test2_features
        gc.collect()

    def plot_class_distribution(self):
        """Plot class distribution comparison between datasets"""
        print("Plotting class distribution comparison...")
        self.setup_plot_style()

        def get_class_distribution(file_path):
            class_counts = {}
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, usecols=['label']):
                chunk_counts = chunk['label'].value_counts()
                for label, count in chunk_counts.items():
                    class_counts[label] = class_counts.get(label, 0) + count
            return pd.Series(class_counts)

        # Get distributions
        test1_dist = get_class_distribution(self.test1_file)
        test2_dist = get_class_distribution(self.test2_file)

        # Ensure the indices are the same
        all_labels = set(test1_dist.index).union(set(test2_dist.index))
        test1_dist = test1_dist.reindex(all_labels, fill_value=0)
        test2_dist = test2_dist.reindex(all_labels, fill_value=0)

        # Plot comparison
        plt.figure(figsize=(15, 8))
        plt.scatter(test1_dist.values, test2_dist.values, alpha=0.6, color='#2ca02c')

        # Add diagonal line
        max_val = max(test1_dist.max(), test2_dist.max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Equal Distribution')

        plt.xlabel('Sample Count (Test Set 1)')
        plt.ylabel('Sample Count (Test Set 2)')
        plt.title('Class Distribution Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig('class_distribution_comparison.png')
        plt.close()

        # Clear memory
        gc.collect()

    def create_tsne_visualization(self, n_samples=5000):
        """Create t-SNE visualization of features"""
        print("Creating t-SNE visualization...")
        self.setup_plot_style()

        # Load samples from both test sets
        test1_data = self.load_data_chunk(self.test1_file, nrows=n_samples)
        test2_data = self.load_data_chunk(self.test2_file, nrows=n_samples)

        # Combine samples and prepare features
        feature_cols = [col for col in test1_data.columns if col not in ['path', 'label']]
        combined_features = pd.concat([test1_data[feature_cols], test2_data[feature_cols]], ignore_index=True)

        # Scale features
        self.scaler.fit(combined_features)
        combined_features_scaled = self.scaler.transform(combined_features)

        # Apply PCA for dimensionality reduction before t-SNE
        pca = PCA(n_components=50, random_state=42)
        combined_features_pca = pca.fit_transform(combined_features_scaled)

        # Create labels for coloring (convert to categorical strings)
        labels = np.concatenate([
            np.full(len(test1_data), 'Test Set 1'),
            np.full(len(test2_data), 'Test Set 2')
        ])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
        reduced_features = tsne.fit_transform(combined_features_pca)

        # Create plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            hue=labels,
            palette={"Test Set 1": "#1f77b4", "Test Set 2": "#ff7f0e"},
            alpha=0.7,
            edgecolor='k',
            s=100
        )

        plt.title('t-SNE Visualization of Test Sets')
        plt.legend(title='Dataset', loc='best')
        plt.savefig('tsne_visualization.png')
        plt.close()

        # Clear memory
        del test1_data, test2_data, combined_features, combined_features_scaled, combined_features_pca
        gc.collect()


    def create_umap_visualization(self, n_samples=5000):
        """Create UMAP visualization of features"""
        print("Creating UMAP visualization...")
        self.setup_plot_style()

        # Load samples from both test sets
        test1_data = self.load_data_chunk(self.test1_file, nrows=n_samples)
        test2_data = self.load_data_chunk(self.test2_file, nrows=n_samples)

        # Combine samples and prepare features
        feature_cols = [col for col in test1_data.columns if col not in ['path', 'label']]
        combined_features = pd.concat([test1_data[feature_cols], test2_data[feature_cols]], ignore_index=True)

        # Scale features
        self.scaler.fit(combined_features)
        combined_features_scaled = self.scaler.transform(combined_features)

        # Create labels for coloring (convert to categorical strings)
        labels = np.concatenate([
            np.full(len(test1_data), 'Test Set 1'),
            np.full(len(test2_data), 'Test Set 2')
        ])

        # Apply UMAP
        reducer = umap.UMAP(random_state=42)
        reduced_features = reducer.fit_transform(combined_features_scaled)

        # Create plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            hue=labels,
            palette={"Test Set 1": "#1f77b4", "Test Set 2": "#ff7f0e"},
            alpha=0.7,
            edgecolor='k',
            s=100
        )

        plt.title('UMAP Visualization of Test Sets')
        plt.legend(title='Dataset', loc='best')
        plt.savefig('umap_visualization.png')
        plt.close()

        # Clear memory
        del test1_data, test2_data, combined_features, combined_features_scaled
        gc.collect()


    def plot_feature_importance(self, model):
        """Create feature importance visualization"""
        print("Plotting feature importance...")
        self.setup_plot_style()

        if hasattr(model, 'coef_'):
            # Get feature importance
            importance = np.abs(model.coef_).mean(axis=0)

            # Load feature names from a small sample
            data_sample = self.load_data_chunk(self.train_file, nrows=1)
            feature_cols = [col for col in data_sample.columns if col not in ['path', 'label']]

            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Plot top 30 features
            plt.figure(figsize=(15, 8))
            sns.barplot(
                data=feature_importance.head(30),
                x='importance',
                y='feature',
                palette='viridis'
            )
            plt.title('Top 30 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()

            # Clear memory
            del data_sample
            gc.collect()
        else:
            print("Model does not have 'coef_' attribute.")

    def plot_confusion_matrices(self, y_true_test1, y_pred_test1, y_true_test2, y_pred_test2):
        """Plot confusion matrices for both test sets"""
        print("Plotting confusion matrices...")
        self.setup_plot_style()

        # Plot for Test Set 1
        cm1 = confusion_matrix(y_true_test1, y_pred_test1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm1, cmap='Blues')
        plt.title('Confusion Matrix - Test Set 1')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_test_set_1.png')
        plt.close()

        # Plot for Test Set 2
        cm2 = confusion_matrix(y_true_test2, y_pred_test2)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm2, cmap='Blues')
        plt.title('Confusion Matrix - Test Set 2')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_test_set_2.png')
        plt.close()

    def plot_class_accuracy_difference(self, y_true_test1, y_pred_test1, y_true_test2, y_pred_test2):
        """Plot difference in class-wise accuracies between test sets"""
        print("Plotting class accuracy difference...")
        self.setup_plot_style()

        # Calculate class-wise accuracies
        from sklearn.metrics import accuracy_score
        import pandas as pd

        df_test1 = pd.DataFrame({'true': y_true_test1, 'pred': y_pred_test1})
        df_test2 = pd.DataFrame({'true': y_true_test2, 'pred': y_pred_test2})

        class_acc_test1 = df_test1.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))
        class_acc_test2 = df_test2.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))

        # Align indices
        all_classes = set(class_acc_test1.index).union(set(class_acc_test2.index))
        class_acc_test1 = class_acc_test1.reindex(all_classes, fill_value=0)
        class_acc_test2 = class_acc_test2.reindex(all_classes, fill_value=0)

        # Calculate difference
        acc_difference = class_acc_test1 - class_acc_test2

        # Plot difference
        plt.figure(figsize=(15, 8))
        sns.barplot(
            x=acc_difference.index,
            y=acc_difference.values,
            palette='coolwarm'
        )
        plt.title('Difference in Class-wise Accuracy (Test Set 1 - Test Set 2)')
        plt.xlabel('Class ID')
        plt.ylabel('Accuracy Difference')
        plt.tight_layout()
        plt.savefig('class_accuracy_difference.png')
        plt.close()

        # Clear memory
        del df_test1, df_test2
        gc.collect()

def main():
    print("Initializing ImageNet visualization pipeline...")
    visualizer = ImageNetVisualizer(chunk_size=5000)

    # Create visualizations
    visualizer.plot_feature_distributions_comparison()
    visualizer.plot_class_distribution()
    visualizer.create_tsne_visualization()
    visualizer.create_umap_visualization()

    print("Visualization pipeline complete! Check the output directory for generated visualizations.")

if __name__ == "__main__":
    main()
