"""
very resource-efficient but time consumming version of train.py with b
BiLSTM with Attention for HDFS Log Anomaly Detection
Handles variable-length sequences with memory-efficient batching
https://claude.ai/chat/3c90f7c2-0eb3-421a-b5e8-b2f023977044 by cloude.ai
"""
"""
Enhanced BiLSTM with Attention for HDFS Log Anomaly Detection
Addresses class imbalance, proper metrics, robust error handling, and anomaly detection best practices

        'epochs': 7,  # Reduced for initial testing intiall was 30

"""
"""
changes made to original code:
    removed -        self.embedding_dim = sequences[0].shape[1] if sequences else 768
    added   +        self.embedding_dim = sequences[0].shape[1] if len(sequences) > 0 else 768
    REMOVED -        test_sequences, test_labels,
    ADDED   +        list(test_sequences), list(test_labels),
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics as keras_metrics
from tensorflow.keras.utils import Sequence
import os
import logging
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           precision_recall_curve, roc_curve, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import gc
import warnings
from typing import List, Tuple, Dict, Optional, Union
import psutil
from collections import Counter
import pickle


# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')


class DataValidator:
    """Validates input data format and quality"""
    
    @staticmethod
    def validate_sequences(sequences: np.ndarray, labels: np.ndarray, 
                          expected_embedding_dim: int = 768) -> Dict:
        """Comprehensive validation of sequence data"""
        validation_report = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Basic structure validation
        if len(sequences) != len(labels):
            validation_report['issues'].append(
                f"Sequence count ({len(sequences)}) != label count ({len(labels)})"
            )
            validation_report['is_valid'] = False
        
        # Sequence format validation
        invalid_sequences = 0
        sequence_lengths = []
        embedding_dims = set()
        
        for i, seq in enumerate(sequences):
            if not isinstance(seq, np.ndarray):
                validation_report['issues'].append(f"Sequence {i} is not numpy array")
                invalid_sequences += 1
                continue
                
            if len(seq.shape) != 2:
                validation_report['issues'].append(f"Sequence {i} has shape {seq.shape}, expected 2D")
                invalid_sequences += 1
                continue
                
            sequence_lengths.append(seq.shape[0])
            embedding_dims.add(seq.shape[1])
            
            # Check for NaN or infinite values
            if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                validation_report['issues'].append(f"Sequence {i} contains NaN/Inf values")
                invalid_sequences += 1
        
        # Embedding dimension consistency
        if len(embedding_dims) > 1:
            validation_report['issues'].append(
                f"Inconsistent embedding dimensions: {embedding_dims}"
            )
            validation_report['is_valid'] = False
        elif len(embedding_dims) == 1:
            dim = list(embedding_dims)[0]
            if dim != expected_embedding_dim:
                validation_report['issues'].append(
                    f"Unexpected embedding dimension: {dim}, expected {expected_embedding_dim}"
                )
        
        # Label validation
        unique_labels = set()
        for label in labels:
            if not isinstance(label, np.ndarray) or len(label) != 2:
                validation_report['issues'].append("Invalid label format")
                validation_report['is_valid'] = False
                break
            unique_labels.add(tuple(label))
        
        # Statistics
        validation_report['statistics'] = {
            'total_sequences': len(sequences),
            'invalid_sequences': invalid_sequences,
            'sequence_lengths': {
                'min': min(sequence_lengths) if sequence_lengths else 0,
                'max': max(sequence_lengths) if sequence_lengths else 0,
                'mean': np.mean(sequence_lengths) if sequence_lengths else 0,
                'std': np.std(sequence_lengths) if sequence_lengths else 0
            },
            'embedding_dims': list(embedding_dims),
            'unique_label_patterns': len(unique_labels)
        }
        
        return validation_report


class MemoryManager:
    """Advanced memory management and monitoring"""
    
    def __init__(self, logger):
        self.logger = logger
        self.peak_memory = 0
        
    def get_memory_info(self):
        """Get comprehensive memory information"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        gpu_memory = None
        if tf.config.list_physical_devices('GPU'):
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            except:
                pass
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'gpu_current_mb': gpu_memory['current'] / 1024 / 1024 if gpu_memory else None,
            'gpu_peak_mb': gpu_memory['peak'] / 1024 / 1024 if gpu_memory else None
        }
    
    def log_memory(self, step=""):
        """Log memory usage with peak tracking"""
        memory_info = self.get_memory_info()
        current_memory = memory_info['rss_mb']
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        log_msg = f"Memory {step}: {current_memory:.1f} MB (Peak: {self.peak_memory:.1f} MB)"
        if memory_info['gpu_current_mb']:
            log_msg += f" | GPU: {memory_info['gpu_current_mb']:.1f} MB"
        
        self.logger.info(log_msg)
        
        # Warning if memory usage is high
        if current_memory > 8000:  # 8GB
            self.logger.warning(f"High memory usage detected: {current_memory:.1f} MB")
    
    def cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            try:
                tf.keras.backend.clear_session()
            except:
                pass  


class AdvancedSequenceDataGenerator(Sequence):
    """
    Enhanced data generator with proper error handling and validation
    """
    
    def __init__(self, sequences, labels, batch_size=32, max_length=None, 
                 shuffle=True, class_weights=None, augment=False, logger=None):
        """
        Args:
            sequences: List of numpy arrays with shape (seq_len, embedding_dim)
            labels: List of one-hot encoded labels
            batch_size: Batch size for training
            max_length: Maximum sequence length (None for dynamic)
            shuffle: Whether to shuffle data between epochs
            class_weights: Dictionary of class weights for handling imbalance
            augment: Whether to apply data augmentation
            logger: Logger instance
        """
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights or {}
        self.augment = augment
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate inputs
        if len(sequences) != len(labels):
            raise ValueError(f"Sequences ({len(sequences)}) and labels ({len(labels)}) must have same length")
        
        # Calculate statistics
        self.sequence_lengths = [len(seq) for seq in sequences]
        self.max_length = max_length or max(self.sequence_lengths)
        self.embedding_dim = sequences[0].shape[1] if len(sequences) > 0 else 768
        
        # Filter sequences by max length
        valid_indices = [i for i, seq_len in enumerate(self.sequence_lengths) 
                        if seq_len <= self.max_length and seq_len > 0]
        
        if len(valid_indices) < len(sequences):
            filtered_count = len(sequences) - len(valid_indices)
            self.logger.warning(f"Filtered out {filtered_count} sequences exceeding max_length={self.max_length}")
        
        self.sequences = [sequences[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]
        self.sequence_lengths = [self.sequence_lengths[i] for i in valid_indices]
        
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()
        
        # Log statistics
        self.logger.info(f"DataGenerator: {len(self.sequences)} sequences, "
                        f"max_len={self.max_length}, embedding_dim={self.embedding_dim}")
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data with error handling"""
        try:
            # Get batch indices
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Generate batch
            return self._generate_batch(batch_indices)
        except Exception as e:
            self.logger.error(f"Error generating batch {index}: {str(e)}")
            # Return empty batch as fallback
            return (np.zeros((1, self.max_length, self.embedding_dim)), 
                   np.zeros((1, 2)))
    
    def on_epoch_end(self):
        """Update indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, batch_indices):
        """Generate batch with fixed padding to max_length and sample weights"""
        batch_size = len(batch_indices)
        
        # Initialize batch arrays with fixed max_length
        X = np.zeros((batch_size, self.max_length, self.embedding_dim), dtype=np.float32)
        y = np.zeros((batch_size, 2), dtype=np.float32)
        sample_weights = np.ones(batch_size, dtype=np.float32)
        
        # Fill batch
        for i, idx in enumerate(batch_indices):
            seq = self.sequences[idx]
            seq_len = len(seq)
            
            # Handle sequence data - pad with zeros to max_length
            X[i, :seq_len, :] = seq.astype(np.float32)
            y[i] = self.labels[idx].astype(np.float32)
            
            # Apply class weights
            class_idx = np.argmax(self.labels[idx])
            sample_weights[i] = self.class_weights.get(class_idx, 1.0)
            
            # Simple data augmentation for anomalies
            if self.augment and class_idx == 1:  # Anomaly class
                X[i] = self._augment_sequence(X[i], seq_len)
        
        return X, y, sample_weights
    
    def _augment_sequence(self, sequence, seq_len):
        """Simple data augmentation for anomaly sequences"""
        # Add small random noise
        noise = np.random.normal(0, 0.01, sequence.shape)
        sequence[:seq_len] += noise[:seq_len]
        return sequence
    
    def get_statistics(self):
        """Get generator statistics"""
        label_counts = Counter(np.argmax(label) for label in self.labels)
        return {
            'total_sequences': len(self.sequences),
            'sequence_lengths': {
                'min': min(self.sequence_lengths),
                'max': max(self.sequence_lengths),
                'mean': np.mean(self.sequence_lengths),
                'median': np.median(self.sequence_lengths)
            },
            'class_distribution': dict(label_counts),
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'batches_per_epoch': len(self)
        }


class RobustAttentionLayer(layers.Layer):
    """
    Improved attention layer with better numerical stability and masking
    """
    
    def __init__(self, attention_dim=64, **kwargs):
        super(RobustAttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.supports_masking = True
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(RobustAttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, timesteps, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Improved masking
        if mask is not None:
            # Convert boolean mask to float and expand dimensions
            mask = tf.cast(mask, tf.float32)
            # Apply large negative value to masked positions
            ait = ait + (1.0 - mask) * (-1e15)
        
        # Compute attention weights with numerical stability
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Ensure weights sum to 1 for numerical stability
        attention_weights = tf.clip_by_value(attention_weights, 1e-15, 1.0)
        attention_weights = attention_weights / tf.reduce_sum(
            attention_weights, axis=1, keepdims=True
        )
        
        # Apply attention weights
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(RobustAttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in anomaly detection
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Ensure numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha weighting
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Combine all components
        focal_loss = alpha_t * focal_weight * ce
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


class AnomalyMetrics:
    """Custom metrics for anomaly detection evaluation"""
    
    @staticmethod
    def compute_threshold_metrics(y_true, y_pred_proba, thresholds=None):
        """Compute metrics at different thresholds"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        results = []
        y_true_binary = np.argmax(y_true, axis=1)
        y_proba_anomaly = y_pred_proba[:, 1]  # Probability of anomaly class
        
        for threshold in thresholds:
            y_pred_binary = (y_proba_anomaly >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })
        
        return results
    
    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
        """Find optimal threshold based on specified metric"""
        threshold_results = AnomalyMetrics.compute_threshold_metrics(y_true, y_pred_proba)
        
        if metric == 'f1':
            best_result = max(threshold_results, key=lambda x: x['f1'])
        elif metric == 'precision':
            best_result = max(threshold_results, key=lambda x: x['precision'])
        elif metric == 'recall':
            best_result = max(threshold_results, key=lambda x: x['recall'])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return best_result


class EnhancedBiLSTMModel:
    """
    Enhanced BiLSTM model with proper anomaly detection practices
    """
    
    def __init__(self, 
                 embedding_dim=768,
                 lstm_units=64,  # Reduced for efficiency
                 attention_dim=32,  # Reduced for efficiency
                 dropout_rate=0.4,  # Increased for regularization
                 learning_rate=0.001,
                 use_focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.model = None
        
    def build_model(self, max_sequence_length=None, class_weights=None):
        """Build the enhanced BiLSTM + Attention model"""
        
        # Input layer
        inputs = layers.Input(
            shape=(max_sequence_length, self.embedding_dim), 
            name='sequence_input'
        )
        
        # Masking layer with explicit mask value
        masked_input = layers.Masking(mask_value=0.0, name='masking')(inputs)
        
        # Single BiLSTM layer (simpler architecture)
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units, 
                return_sequences=True, 
                dropout=self.dropout_rate,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            name='bilstm'
        )(masked_input)
        
        # Batch normalization
        lstm_out = layers.BatchNormalization(name='batch_norm')(lstm_out)
        
        # Attention mechanism
        attention_output = RobustAttentionLayer(
            attention_dim=self.attention_dim,
            name='attention'
        )(lstm_out)

        # Remove mask after attention to prevent dimension mismatch in subsequent layers
        attention_output = layers.Lambda(lambda x: x, name='remove_mask')(attention_output)

        # Dense layers with regularization
        dense1 = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense_1'
        )(attention_output)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        dense1 = layers.BatchNormalization()(dense1)
        
        # Output layer - using sigmoid for binary classification
        outputs = layers.Dense(2, activation='softmax', name='output')(dense1)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='Enhanced_BiLSTM_Attention')
        
        # Choose loss function
        if self.use_focal_loss:
            loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = 'categorical_crossentropy'
        
        # Custom metrics for anomaly detection
        custom_metrics = [
            'accuracy',
            keras_metrics.Precision(name='precision'),
            keras_metrics.Recall(name='recall'),
            keras_metrics.AUC(name='auc_roc'),
            keras_metrics.AUC(name='auc_pr', curve='PR')
        ]
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=custom_metrics
        )
        
        return self.model


class HDFSAnomalyTrainer:
    """
    Enhanced trainer with proper anomaly detection practices
    """
    
    def __init__(self, data_dir='preprocessed_data_2k_new', output_dir='model_output'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.memory_manager = MemoryManager(self.logger)
        self.model_wrapper = None
        self.class_weights = None
        self.optimal_threshold = 0.5
        
        # Data generators
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        # Training history and results
        self.training_history = {}
        self.evaluation_results = {}
        
    def setup_logging(self):
        """Enhanced logging setup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f'training_{timestamp}.log'

        # Configure logging with UTF-8 encoding for console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced HDFS Anomaly Detection Training Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def load_and_validate_data(self, max_sequence_length=100):
        """Load data with comprehensive validation"""
        self.logger.info("=== Loading and Validating Data ===")
        self.memory_manager.log_memory("before data loading")
        
        # Check file existence
        train_path = self.data_dir / 'training_data.npz'
        test_path = self.data_dir / 'testing_data.npz'
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        # Load data
        try:
            train_data = np.load(train_path, allow_pickle=True)
            test_data = np.load(test_path, allow_pickle=True)
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
        
        # Extract sequences and labels
        train_sequences = train_data['x']
        train_labels = train_data['y']
        test_sequences = test_data['x']
        test_labels = test_data['y']
        
        self.logger.info(f"Raw data loaded: {len(train_sequences)} train, {len(test_sequences)} test")
        
        # Validate data
        self.logger.info("Validating training data...")
        train_validation = DataValidator.validate_sequences(train_sequences, train_labels)
        
        if not train_validation['is_valid']:
            self.logger.error("Training data validation failed:")
            for issue in train_validation['issues']:
                self.logger.error(f"  - {issue}")
            raise ValueError("Training data validation failed")
        
        self.logger.info("Validating test data...")
        test_validation = DataValidator.validate_sequences(test_sequences, test_labels)
        
        if not test_validation['is_valid']:
            self.logger.error("Test data validation failed:")
            for issue in test_validation['issues']:
                self.logger.error(f"  - {issue}")
            # Continue with warning for test data
        
        # Log validation statistics
        self.logger.info("=== Data Statistics ===")
        train_stats = train_validation['statistics']
        test_stats = test_validation['statistics']
        
        self.logger.info(f"Training sequences: {train_stats['total_sequences']}")
        self.logger.info(f"Training seq lengths - Min: {train_stats['sequence_lengths']['min']}, "
                        f"Max: {train_stats['sequence_lengths']['max']}, "
                        f"Mean: {train_stats['sequence_lengths']['mean']:.1f}")
        
        self.logger.info(f"Test sequences: {test_stats['total_sequences']}")
        self.logger.info(f"Test seq lengths - Min: {test_stats['sequence_lengths']['min']}, "
                        f"Max: {test_stats['sequence_lengths']['max']}, "
                        f"Mean: {test_stats['sequence_lengths']['mean']:.1f}")
        
        return (train_sequences, train_labels, test_sequences, test_labels, 
                train_stats, test_stats)
    
    def prepare_data_generators(self, train_sequences, train_labels, test_sequences, test_labels,
                               validation_split=0.2, max_sequence_length=100, batch_size=32):
        """Prepare data generators with class balancing"""
        
        # Calculate class weights
        y_integers = np.argmax(train_labels, axis=1)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=np.unique(y_integers), 
            y=y_integers
        )
        self.class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        self.logger.info(f"Class weights: {self.class_weights}")
        
        # Log class distribution before split
        train_class_counts = Counter(y_integers)
        self.logger.info(f"Training class distribution: {dict(train_class_counts)}")
        
        # Create stratified train/validation split
        try:
            train_idx, val_idx = train_test_split(
                range(len(train_sequences)), 
                test_size=validation_split,
                stratify=y_integers,
                random_state=42
            )
        except ValueError as e:
            self.logger.warning(f"Stratified split failed: {e}. Using random split.")
            train_idx, val_idx = train_test_split(
                range(len(train_sequences)),
                test_size=validation_split,
                random_state=42
            )
        
        # Split data
        train_seqs = [train_sequences[i] for i in train_idx]
        train_labs = [train_labels[i] for i in train_idx]
        val_seqs = [train_sequences[i] for i in val_idx]
        val_labs = [train_labels[i] for i in val_idx]
        
        # Log final distributions
        self.log_data_distribution(train_labs, val_labs, test_labels)
        
        # Create generators
        self.train_generator = AdvancedSequenceDataGenerator(
            train_seqs, train_labs,
            batch_size=batch_size,
            max_length=max_sequence_length,
            shuffle=True,
            class_weights=self.class_weights,
            augment=True,  # Enable augmentation for training
            logger=self.logger
        )
        
        self.val_generator = AdvancedSequenceDataGenerator(
            val_seqs, val_labs,
            batch_size=batch_size,
            max_length=max_sequence_length,
            shuffle=False,
            class_weights=self.class_weights,
            augment=False,
            logger=self.logger
        )
        
        self.test_generator = AdvancedSequenceDataGenerator(
            list(test_sequences), list(test_labels),
            batch_size=batch_size,
            max_length=max_sequence_length,
            shuffle=False,
            class_weights=self.class_weights,
            augment=False,
            logger=self.logger
        )
        
        # Log generator statistics
        self.logger.info("=== Generator Statistics ===")
        train_stats = self.train_generator.get_statistics()
        val_stats = self.val_generator.get_statistics()
        test_stats = self.test_generator.get_statistics()
        
        self.logger.info(f"Training generator: {train_stats}")
        self.logger.info(f"Validation generator: {val_stats}")
        self.logger.info(f"Test generator: {test_stats}")
        
        self.memory_manager.log_memory("after data preparation")
        
        return max_sequence_length
    
    def log_data_distribution(self, train_labels, val_labels, test_labels):
        """Log detailed data distribution"""
        def count_labels(labels):
            counts = Counter(np.argmax(label) for label in labels)
            normal = counts.get(0, 0)
            anomaly = counts.get(1, 0)
            return normal, anomaly
        
        train_normal, train_anomaly = count_labels(train_labels)
        val_normal, val_anomaly = count_labels(val_labels)
        test_normal, test_anomaly = count_labels(test_labels)
        
        total_train = train_normal + train_anomaly
        total_val = val_normal + val_anomaly
        total_test = test_normal + test_anomaly
        
        self.logger.info("=== Final Data Distribution ===")
        self.logger.info(f"Training:   {train_normal:,} Normal ({train_normal/total_train*100:.1f}%), "
                        f"{train_anomaly:,} Anomaly ({train_anomaly/total_train*100:.1f}%)")
        self.logger.info(f"Validation: {val_normal:,} Normal ({val_normal/total_val*100:.1f}%), "
                        f"{val_anomaly:,} Anomaly ({val_anomaly/total_val*100:.1f}%)")
        self.logger.info(f"Testing:    {test_normal:,} Normal ({test_normal/total_test*100:.1f}%), "
                        f"{test_anomaly:,} Anomaly ({test_anomaly/total_test*100:.1f}%)")
        
        # Check for severe imbalance
        test_imbalance_ratio = test_normal / max(test_anomaly, 1)
        if test_imbalance_ratio > 1000:
            self.logger.warning(f"Severe class imbalance detected in test set: "
                              f"{test_imbalance_ratio:.0f}:1 ratio")
    
    def build_model(self, max_sequence_length):
        """Build the enhanced model"""
        self.logger.info("=== Building Enhanced BiLSTM + Attention Model ===")
        
        self.model_wrapper = EnhancedBiLSTMModel(
            embedding_dim=768,
            lstm_units=64,  # Optimized for efficiency
            attention_dim=32,
            dropout_rate=0.4,
            learning_rate=0.001,
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        model = self.model_wrapper.build_model(
            max_sequence_length=max_sequence_length,
            class_weights=self.class_weights
        )
        
        self.logger.info("Model architecture:")
        model.summary(print_fn=self.logger.info)
        
        # Calculate model parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def setup_callbacks(self):
        """Setup comprehensive callbacks"""
        callbacks_list = [
            # Early stopping based on validation AUC
            callbacks.EarlyStopping(
                monitor='val_auc_pr',  # Use AUC-PR for imbalanced data
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            # Model checkpoint - save best model
            callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.h5'),
                monitor='val_auc_pr',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                str(self.output_dir / 'training_log.csv'),
                append=True
            ),
            
            # Custom callback for memory monitoring
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.memory_manager.log_memory(f"epoch {epoch+1}")
            )
        ]
        
        return callbacks_list
    
    def train_model(self, epochs=50):
        """Train the model with proper monitoring"""
        if self.model_wrapper is None or self.train_generator is None:
            raise ValueError("Model and data must be prepared first")
        
        self.logger.info(f"=== Starting Training for {epochs} epochs ===")
        self.memory_manager.log_memory("before training")
        
        # Setup callbacks
        callback_list = self.setup_callbacks()
        
        # Train model
        try:
            history = self.model_wrapper.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callback_list,
                verbose=1
            )
            
            self.training_history = history.history
            self.logger.info("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
        self.memory_manager.log_memory("after training")
        return history
    
    def find_optimal_threshold(self):
        """Find optimal threshold using validation data"""
        self.logger.info("=== Finding Optimal Classification Threshold ===")
        
        # Get predictions on validation set
        val_predictions = self.model_wrapper.model.predict(self.val_generator, verbose=1)
        
        # Get true labels
        val_true_labels = []
        for i in range(len(self.val_generator)):
            _, batch_y, _ = self.val_generator[i]  # Note: includes sample weights
            val_true_labels.extend(batch_y)
        
        val_true_labels = np.array(val_true_labels[:len(val_predictions)])
        
        # Find optimal threshold
        optimal_result = AnomalyMetrics.find_optimal_threshold(
            val_true_labels, val_predictions, metric='f1'
        )
        
        self.optimal_threshold = optimal_result['threshold']
        self.logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        self.logger.info(f"At optimal threshold - F1: {optimal_result['f1']:.3f}, "
                        f"Precision: {optimal_result['precision']:.3f}, "
                        f"Recall: {optimal_result['recall']:.3f}")
        
        return self.optimal_threshold
    
    def comprehensive_evaluation(self):
        """Comprehensive model evaluation"""
        if self.model_wrapper is None or self.test_generator is None:
            raise ValueError("Model and test data must be prepared first")
        
        self.logger.info("=== Comprehensive Model Evaluation ===")
        
        # Basic evaluation
        test_metrics = self.model_wrapper.model.evaluate(self.test_generator, verbose=1)
        metric_names = self.model_wrapper.model.metrics_names
        
        # Log basic metrics
        self.logger.info("Basic Test Metrics:")
        for name, value in zip(metric_names, test_metrics):
            self.logger.info(f"  {name}: {value:.4f}")
        
        # Get predictions and true labels
        test_predictions = self.model_wrapper.model.predict(self.test_generator, verbose=1)
        
        test_true_labels = []
        for i in range(len(self.test_generator)):
            _, batch_y, _ = self.test_generator[i]
            test_true_labels.extend(batch_y)
        
        test_true_labels = np.array(test_true_labels[:len(test_predictions)])
        
        # Convert to binary
        y_true_binary = np.argmax(test_true_labels, axis=1)
        y_proba = test_predictions[:, 1]  # Probability of anomaly class
        
        # Calculate AUC scores
        try:
            auc_roc = roc_auc_score(y_true_binary, y_proba)
            auc_pr = average_precision_score(y_true_binary, y_proba)
        except ValueError as e:
            self.logger.warning(f"Could not calculate AUC scores: {e}")
            auc_roc = auc_pr = 0.0
        
        # Threshold analysis
        threshold_results = AnomalyMetrics.compute_threshold_metrics(
            test_true_labels, test_predictions
        )
        
        # Find best F1 threshold
        best_f1_result = max(threshold_results, key=lambda x: x['f1'])
        
        # Predictions at optimal threshold
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        
        # Classification report
        if len(np.unique(y_true_binary)) > 1:
            report = classification_report(
                y_true_binary, y_pred_optimal,
                target_names=['Normal', 'Anomaly'],
                digits=4,
                output_dict=True
            )
        else:
            report = {"warning": "Only one class present in test set"}
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_optimal)
        
        # Compile results
        self.evaluation_results = {
            'basic_metrics': dict(zip(metric_names, test_metrics)),
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'optimal_threshold': self.optimal_threshold,
            'best_f1_threshold': best_f1_result['threshold'],
            'best_f1_score': best_f1_result['f1'],
            'threshold_analysis': threshold_results,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Log detailed results
        self.logger.info(f"AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"AUC-PR: {auc_pr:.4f}")
        self.logger.info(f"Best F1 Score: {best_f1_result['f1']:.4f} at threshold {best_f1_result['threshold']:.3f}")
        
        if isinstance(report, dict) and 'anomaly' in report:
            self.logger.info(f"Anomaly Class - Precision: {report['Anomaly']['precision']:.4f}, "
                           f"Recall: {report['Anomaly']['recall']:.4f}, "
                           f"F1: {report['Anomaly']['f1-score']:.4f}")
        
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save results
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in self.evaluation_results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to {results_path}")
        
        return self.evaluation_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        self.logger.info("=== Creating Visualizations ===")
        
        plt.style.use('default')
        
        # Training history plots
        if self.training_history:
            self.plot_training_history()
        
        # Threshold analysis plot
        if 'threshold_analysis' in self.evaluation_results:
            self.plot_threshold_analysis()
        
        # Confusion matrix heatmap
        if 'confusion_matrix' in self.evaluation_results:
            self.plot_confusion_matrix()
        
        # ROC and PR curves
        if self.test_generator is not None:
            self.plot_roc_pr_curves()
    
    def plot_training_history(self):
        """Plot training history with all metrics"""
        history = self.training_history
        
        # Determine number of subplots based on available metrics
        available_metrics = [m for m in ['loss', 'accuracy', 'precision', 'recall', 'auc_roc', 'auc_pr'] 
                           if m in history]
        
        n_metrics = len(available_metrics)
        if n_metrics == 0:
            return
        
        # Calculate grid layout
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Plot training and validation curves
            ax.plot(history[metric], label=f'Training {metric.title()}', linewidth=2)
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric.title()}', linewidth=2)
            
            ax.set_title(f'Model {metric.title()}', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history saved to {self.output_dir / 'training_history.png'}")
    
    def plot_threshold_analysis(self):
        """Plot threshold analysis"""
        threshold_results = self.evaluation_results['threshold_analysis']
        
        thresholds = [r['threshold'] for r in threshold_results]
        precision = [r['precision'] for r in threshold_results]
        recall = [r['recall'] for r in threshold_results]
        f1 = [r['f1'] for r in threshold_results]
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(thresholds, precision, 'b-', linewidth=2, label='Precision')
        plt.plot(thresholds, recall, 'r-', linewidth=2, label='Recall')
        plt.plot(thresholds, f1, 'g-', linewidth=2, label='F1-Score')
        
        # Mark optimal threshold
        plt.axvline(x=self.optimal_threshold, color='orange', linestyle='--', 
                   linewidth=2, label=f'Optimal Threshold ({self.optimal_threshold:.3f})')
        
        plt.xlabel('Classification Threshold')
        plt.ylabel('Metric Value')
        plt.title('Threshold Analysis: Precision, Recall, and F1-Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Threshold analysis saved to {self.output_dir / 'threshold_analysis.png'}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        cm = np.array(self.evaluation_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")
    
    def plot_roc_pr_curves(self):
        """Plot ROC and Precision-Recall curves"""
        # Get test predictions
        test_predictions = self.model_wrapper.model.predict(self.test_generator, verbose=0)
        
        # Get true labels
        test_true_labels = []
        for i in range(len(self.test_generator)):
            _, batch_y, _ = self.test_generator[i]
            test_true_labels.extend(batch_y)
        
        test_true_labels = np.array(test_true_labels[:len(test_predictions)])
        y_true_binary = np.argmax(test_true_labels, axis=1)
        y_proba = test_predictions[:, 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
            auc_roc = self.evaluation_results.get('auc_roc', 0)
            
            ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_proba)
            auc_pr = self.evaluation_results.get('auc_pr', 0)
            
            ax2.plot(recall_curve, precision_curve, linewidth=2, 
                    label=f'PR Curve (AUC = {auc_pr:.3f})')
            
            # Baseline (random classifier performance)
            baseline = np.sum(y_true_binary) / len(y_true_binary)
            ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                       label=f'Baseline ({baseline:.3f})')
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Cannot plot ROC curve\nOnly one class in test set',
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Cannot plot PR curve\nOnly one class in test set',
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC and PR curves saved to {self.output_dir / 'roc_pr_curves.png'}")
    
    def save_artifacts(self):
        """Save all model artifacts"""
        self.logger.info("=== Saving Model Artifacts ===")
        
        # Save final model
        if self.model_wrapper and self.model_wrapper.model:
            final_model_path = self.output_dir / 'final_model.h5'
            self.model_wrapper.model.save(str(final_model_path))
            self.logger.info(f"Final model saved to {final_model_path}")
        
        # Save class weights
        class_weights_path = self.output_dir / 'class_weights.json'
        with open(class_weights_path, 'w') as f:
            json.dump(self.class_weights, f, indent=2)
        
        # Save optimal threshold
        threshold_path = self.output_dir / 'optimal_threshold.json'
        with open(threshold_path, 'w') as f:
            json.dump({'optimal_threshold': self.optimal_threshold}, f, indent=2)
        
        # Save training history
        if self.training_history:
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists
                history_json = {}
                for key, value in self.training_history.items():
                    if isinstance(value, np.ndarray):
                        history_json[key] = value.tolist()
                    elif isinstance(value, list):
                        history_json[key] = value
                    else:
                        history_json[key] = str(value)
                json.dump(history_json, f, indent=2)
        
        self.logger.info("All artifacts saved successfully!")
    
    def run_complete_pipeline(self, epochs=50, max_sequence_length=100, 
                            batch_size=32, validation_split=0.2):
        """Run the complete enhanced training pipeline"""
        try:
            self.logger.info("=== Starting Enhanced HDFS Anomaly Detection Pipeline ===")
            
            # Load and validate data
            (train_sequences, train_labels, test_sequences, test_labels, 
             train_stats, test_stats) = self.load_and_validate_data(max_sequence_length)
            
            # Prepare data generators
            max_seq_len = self.prepare_data_generators(
                train_sequences, train_labels, test_sequences, test_labels,
                validation_split=validation_split,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size
            )
            
            # Build model
            self.build_model(max_seq_len)
            
            # Train model
            self.train_model(epochs=epochs)
            
            # Find optimal threshold
            self.find_optimal_threshold()
            
            # Comprehensive evaluation
            self.comprehensive_evaluation()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save artifacts
            self.save_artifacts()
            
            # Cleanup memory
            self.memory_manager.cleanup()
            
            self.logger.info("=== Enhanced Training Pipeline Completed Successfully ===")
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Enhanced training pipeline failed: {str(e)}")
            self.logger.error("Full traceback:", exc_info=True)
            raise


def main():
    """Main execution function with robust error handling"""
    
    # Configuration
    CONFIG = {
        'data_dir': 'preprocessed_data_2k_new',
        'output_dir': 'enhanced_model_output',
        'epochs': 123,  # Reduced for initial testing
        'max_sequence_length': 100,
        'batch_size': 32,  # Reduced for memory efficiency
        'validation_split': 0.2
    }
    
    # Setup GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    else:
        print("No GPUs found, using CPU")
    
    # Initialize trainer
    trainer = HDFSAnomalyTrainer(
        data_dir=CONFIG['data_dir'],
        output_dir=CONFIG['output_dir']
    )
    
    try:
        # Run enhanced training pipeline
        results = trainer.run_complete_pipeline(
            epochs=CONFIG['epochs'],
            max_sequence_length=CONFIG['max_sequence_length'],
            batch_size=CONFIG['batch_size'],
            validation_split=CONFIG['validation_split']
        )
        
        print("\n" + "="*80)
        print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f" Output directory: {CONFIG['output_dir']}/")
        print(f" Best model: best_model.h5")
        print(f" Final model: final_model.h5")
        print(f" Training plots: training_history.png")
        print(f" Threshold analysis: threshold_analysis.png")
        print(f" Confusion matrix: confusion_matrix.png")
        print(f" ROC/PR curves: roc_pr_curves.png")
        print(f" Detailed results: evaluation_results.json")
        print(f" Training logs: training_*.log")
        print("="*80)
        
        # Print key results
        if results:
            print(" KEY RESULTS:")
            print(f"   AUC-ROC: {results.get('auc_roc', 0):.4f}")
            print(f"   AUC-PR:  {results.get('auc_pr', 0):.4f}")
            print(f"   Best F1: {results.get('best_f1_score', 0):.4f}")
            print(f"   Optimal Threshold: {results.get('optimal_threshold', 0.5):.3f}")
        
        return 0
        
    except Exception as e:
        print(f"\n Training failed: {str(e)}")
        print(f"Check the log files in {CONFIG['output_dir']}/ for details")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())