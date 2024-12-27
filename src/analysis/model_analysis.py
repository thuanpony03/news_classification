import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from typing import Dict, List, Tuple
import logging


class ModelAnalyzer:
    def __init__(
            self,
            model: tf.keras.Model,
            tokenizer,
            label_encoder,
            config: Dict
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_predictions(
            self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            texts: List[str]
    ) -> pd.DataFrame:
        """Phân tích chi tiết các dự đoán"""
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Create analysis DataFrame
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': self.label_encoder.inverse_transform(y_test),
            'predicted_label': self.label_encoder.inverse_transform(y_pred),
            'confidence': np.max(y_pred_proba, axis=1)
        })

        # Add prediction status
        results_df['correct'] = results_df['true_label'] == results_df['predicted_label']

        # Add top 3 predictions with probabilities
        top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
        top_k_probs = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]

        for i in range(3):
            results_df[f'top_{i + 1}_prediction'] = self.label_encoder.inverse_transform(
                top_k_indices[:, i]
            )
            results_df[f'top_{i + 1}_probability'] = top_k_probs[:, i]

        return results_df

    def plot_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            figsize: Tuple[int, int] = (12, 8)
    ):
        """Vẽ confusion matrix"""
        plt.figure(figsize=figsize)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        return plt

    def analyze_errors(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Phân tích các trường hợp dự đoán sai"""
        error_df = results_df[~results_df['correct']].copy()
        error_df['confidence_diff'] = error_df.apply(
            lambda x: x[f'top_1_probability'] - x[f'top_2_probability'],
            axis=1
        )
        return error_df.sort_values('confidence_diff')

    def analyze_attention_weights(
            self,
            text: str,
            layer_name: str = 'attention'
    ) -> np.ndarray:
        """Phân tích attention weights cho một văn bản cụ thể"""
        # Tokenize and pad text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.config['max_sequence_length']
        )

        # Create attention model
        attention_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )

        # Get attention weights
        attention_weights = attention_model.predict(padded_sequence)

        return attention_weights

    def visualize_attention(
            self,
            text: str,
            attention_weights: np.ndarray,
            figsize: Tuple[int, int] = (15, 5)
    ):
        """Visualize attention weights"""
        # Tokenize text
        tokens = self.tokenizer.texts_to_sequences([text])[0]
        words = [self.tokenizer.index_word.get(token, '') for token in tokens]

        # Plot attention weights
        plt.figure(figsize=figsize)
        sns.heatmap(
            attention_weights,
            xticklabels=words,
            yticklabels=False,
            cmap='viridis'
        )
        plt.title('Attention Weights Visualization')
        plt.xlabel('Words')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt

    def generate_analysis_report(
            self,
            results_df: pd.DataFrame,
            output_path: str
    ):
        """Tạo báo cáo phân tích tổng hợp"""
        report = []

        # Overall metrics
        accuracy = results_df['correct'].mean()
        report.append(f"Overall Accuracy: {accuracy:.4f}")

        # Per-class metrics
        class_metrics = results_df.groupby('true_label')['correct'].agg(['mean', 'count'])
        report.append("\nPer-class Performance:")
        for idx, row in class_metrics.iterrows():
            report.append(f"{idx}:")
            report.append(f"  Accuracy: {row['mean']:.4f}")
            report.append(f"  Samples: {row['count']}")

        # Confidence analysis
        report.append("\nConfidence Analysis:")
        report.append(f"Mean confidence (correct): {results_df[results_df['correct']]['confidence'].mean():.4f}")
        report.append(f"Mean confidence (incorrect): {results_df[~results_df['correct']]['confidence'].mean():.4f}")

        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        return '\n'.join(report)