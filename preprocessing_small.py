# Enhanced Log Preprocessing for Anomaly Detection with Improved Error Handling and Memory Efficiency

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import warnings
import json
from pathlib import Path


@dataclass
class PreprocessConfig:
    """Configuration class for preprocessing parameters"""
    model_name: str = 'distilbert-base-nli-mean-tokens'
    batch_size: int = 32
    num_workers: int = 6
    train_size: float = 0.7
    test_size: float = 0.3
    min_test_anomalies: int = 10
    max_block_size: int = 1000  # Limit block size for memory efficiency
    embedding_dim: int = 768
    blockid_pattern: str = r'blk_-?\d+'
    output_dir: str = 'preprocessed_data_2k_new'
    log_level: str = 'INFO'
    random_state: int = 42


class LogPreprocessor:
    """Enhanced log preprocessing class for anomaly detection"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.template_cache = {}
        self.stats = {
            'total_logs': 0,
            'processed_logs': 0,
            'dropped_logs': 0,
            'unique_blocks': 0,
            'unknown_templates': 0,
            'data_quality_issues': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_model(self) -> None:
        """Load sentence transformer model with error handling"""
        try:
            self.logger.info(f"Loading model {self.config.model_name} on {self.device}")
            self.model = SentenceTransformer(self.config.model_name, device=self.device)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _validate_input_files(self, file_paths: List[str]) -> bool:
        """Validate that input files exist and are readable"""
        for file_path in file_paths:
            if not Path(file_path).exists():
                self.logger.error(f"Input file not found: {file_path}")
                return False
            try:
                pd.read_csv(file_path, nrows=1)
            except Exception as e:
                self.logger.error(f"Cannot read file {file_path}: {e}")
                return False
        return True
    
    def extract_blockid_from_content(self, content_series: pd.Series) -> Tuple[List[str], Dict]:
        """Enhanced BlockID extraction with validation and statistics"""
        block_pattern = re.compile(self.config.blockid_pattern)
        block_ids = []
        extraction_stats = {
            'total': len(content_series),
            'extracted': 0,
            'missing': 0,
            'invalid_format': 0
        }
        
        for content in tqdm(content_series, desc='Extracting BlockIDs'):
            if pd.isna(content):
                block_ids.append(None)
                extraction_stats['missing'] += 1
                continue
            
            matches = block_pattern.findall(str(content))
            if matches:
                # Take the first match, validate format
                block_id = matches[0]
                if self._validate_blockid_format(block_id):
                    block_ids.append(block_id)
                    extraction_stats['extracted'] += 1
                else:
                    block_ids.append(None)
                    extraction_stats['invalid_format'] += 1
            else:
                block_ids.append(None)
                extraction_stats['missing'] += 1
        
        # Log extraction statistics
        success_rate = extraction_stats['extracted'] / extraction_stats['total'] * 100
        self.logger.info(f"BlockID extraction: {success_rate:.1f}% success rate")
        self.logger.info(f"Extracted: {extraction_stats['extracted']}, "
                        f"Missing: {extraction_stats['missing']}, "
                        f"Invalid: {extraction_stats['invalid_format']}")
        
        if success_rate < 50:
            self.logger.warning("Low BlockID extraction rate - check regex pattern")
            self.stats['data_quality_issues'].append("Low BlockID extraction rate")
        
        return block_ids, extraction_stats
    
    def _validate_blockid_format(self, block_id: str) -> bool:
        """Validate BlockID format"""
        # Add more sophisticated validation if needed
        return bool(re.match(r'^blk_-?\d+$', block_id))
    
    def _encode_templates_batch(self, templates: List[str]) -> np.ndarray:
        """Encode templates in batches for memory efficiency"""
        if not templates:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                templates, 
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Validate embedding dimensions
            if embeddings.shape[1] != self.config.embedding_dim:
                self.logger.warning(f"Unexpected embedding dimension: {embeddings.shape[1]}")
            
            return embeddings
        except Exception as e:
            self.logger.error(f"Error encoding templates: {e}")
            raise
    
    def create_template_cache(self, df_template: pd.DataFrame) -> None:
        """Create template to vector cache with batch processing"""
        self.logger.info("Creating template cache...")
        
        if 'EventTemplate' not in df_template.columns:
            raise ValueError("EventTemplate column not found in template data")
        
        templates = df_template['EventTemplate'].dropna().tolist()
        if not templates:
            raise ValueError("No valid templates found")
        
        embeddings = self._encode_templates_batch(templates)
        
        # Create cache
        self.template_cache = dict(zip(templates, embeddings))
        self.logger.info(f"Created template cache with {len(self.template_cache)} templates")
    
    def convert_templates_to_vectors(self, df_structured: pd.DataFrame) -> pd.DataFrame:
        """Convert templates to vectors with improved error handling"""
        if 'EventTemplate' not in df_structured.columns:
            raise ValueError("EventTemplate column not found in structured data")
        
        vectors = []
        unknown_templates = []
        
        self.logger.info("Converting templates to vectors...")
        
        for template in tqdm(df_structured['EventTemplate'], desc='Template conversion'):
            if pd.isna(template):
                # Handle missing templates with zero vector
                vectors.append(np.zeros(self.config.embedding_dim))
                continue
                
            if template in self.template_cache:
                vectors.append(self.template_cache[template])
            else:
                # Handle unknown templates
                unknown_templates.append(template)
                try:
                    vector = self.model.encode([str(template)])[0]
                    vectors.append(vector)
                    self.template_cache[template] = vector
                except Exception as e:
                    self.logger.warning(f"Failed to encode template '{template}': {e}")
                    vectors.append(np.zeros(self.config.embedding_dim))
        
        df_result = df_structured.copy()
        df_result['Vector'] = vectors
        
        self.stats['unknown_templates'] = len(unknown_templates)
        if unknown_templates:
            self.logger.info(f"Processed {len(unknown_templates)} unknown templates")
            
        return df_result
    
    def _create_balanced_split(self, df_label: pd.DataFrame, available_blocks: set) -> pd.DataFrame:
        """Create balanced train/test split with proper stratification"""
        # Filter labels to only include available blocks
        df_filtered = df_label[df_label['BlockId'].isin(available_blocks)].copy()
        
        if df_filtered.empty:
            raise ValueError("No common blocks found between structured and label data")
        
        # Separate normal and anomaly blocks
        normal_blocks = df_filtered[df_filtered['Label'] == 'Normal']
        anomaly_blocks = df_filtered[df_filtered['Label'] == 'Anomaly']
        
        self.logger.info(f"Available blocks - Normal: {len(normal_blocks)}, Anomaly: {len(anomaly_blocks)}")
        
        # Ensure minimum test set size for anomalies
        min_test_anomalies = min(self.config.min_test_anomalies, len(anomaly_blocks) // 4)
        
        train_indices = []
        
        # Split anomaly blocks
        if len(anomaly_blocks) > min_test_anomalies:
            anomaly_train, _ = train_test_split(
                anomaly_blocks,
                train_size=len(anomaly_blocks) - min_test_anomalies,
                random_state=self.config.random_state,
                stratify=None
            )
            train_indices.extend(anomaly_train.index)
        
        # Split normal blocks
        if len(normal_blocks) > 0:
            train_size = min(int(len(normal_blocks) * self.config.train_size), 
                           len(normal_blocks) - 100)  # Leave at least 100 for testing
            if train_size > 0:
                normal_train, _ = train_test_split(
                    normal_blocks,
                    train_size=train_size,
                    random_state=self.config.random_state
                )
                train_indices.extend(normal_train.index)
        
        # Set usage labels
        df_result = df_filtered.copy()
        df_result['Usage'] = 'testing'
        df_result.loc[train_indices, 'Usage'] = 'training'
        
        # Log split statistics
        train_normal = len(df_result[(df_result['Usage'] == 'training') & (df_result['Label'] == 'Normal')])
        train_anomaly = len(df_result[(df_result['Usage'] == 'training') & (df_result['Label'] == 'Anomaly')])
        test_normal = len(df_result[(df_result['Usage'] == 'testing') & (df_result['Label'] == 'Normal')])
        test_anomaly = len(df_result[(df_result['Usage'] == 'testing') & (df_result['Label'] == 'Anomaly')])
        
        self.logger.info(f"Data split - Training: {train_normal} Normal, {train_anomaly} Anomaly")
        self.logger.info(f"Data split - Testing: {test_normal} Normal, {test_anomaly} Anomaly")
        
        return df_result
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> bool:
        """Validate data consistency and quality"""
        issues = []
        
        # Check for duplicate BlockIDs across train/test
        train_blocks = set(df[df['Usage'] == 'training']['BlockId'].unique())
        test_blocks = set(df[df['Usage'] == 'testing']['BlockId'].unique())
        overlap = train_blocks.intersection(test_blocks)
        
        if overlap:
            issues.append(f"Data leakage: {len(overlap)} blocks in both train and test")
        
        # Check vector dimensions
        if 'Vector' in df.columns:
            sample_vectors = df['Vector'].dropna().head(100)
            if not sample_vectors.empty:
                dims = [len(v) for v in sample_vectors if hasattr(v, '__len__')]
                if dims and (min(dims) != max(dims) or dims[0] != self.config.embedding_dim):
                    issues.append(f"Inconsistent vector dimensions: {set(dims)}")
        
        # Check for missing critical data
        missing_vectors = df['Vector'].isna().sum()
        if missing_vectors > 0:
            issues.append(f"{missing_vectors} missing vectors")
        
        if issues:
            for issue in issues:
                self.logger.error(f"Data consistency issue: {issue}")
            self.stats['data_quality_issues'].extend(issues)
            return False
        
        return True
    
    def preprocess_data_blocks(self, df: pd.DataFrame, mode: str) -> None:
        """Enhanced preprocessing with memory-efficient block processing"""
        if df.empty:
            self.logger.warning(f"No data available for {mode} preprocessing")
            return
        
        x_data, y_data, block_info = [], [], []
        
        # Group by BlockId and process in chunks
        grouped = df.groupby('BlockId')
        self.logger.info(f"Processing {len(grouped)} blocks for {mode}")
        
        for blk_id, group in tqdm(grouped, desc=f'Processing {mode} blocks'):
            if pd.isna(blk_id):
                continue
            
            # Limit block size to prevent memory issues
            if len(group) > self.config.max_block_size:
                self.logger.warning(f"Block {blk_id} too large ({len(group)} entries), truncating")
                group = group.head(self.config.max_block_size)
            
            # Convert vectors to numpy array
            try:
                vectors = np.stack(group['Vector'].tolist())
            except Exception as e:
                self.logger.error(f"Error processing block {blk_id}: {e}")
                continue
            
            # Validate vector shape
            if vectors.shape[1] != self.config.embedding_dim:
                self.logger.warning(f"Block {blk_id} has incorrect vector dimension")
                continue
            
            x_data.append(vectors)
            
            # Create one-hot encoded label
            label = group.iloc[0]['Label']
            y_index = 1 if label == 'Anomaly' else 0
            y = [0, 0]
            y[y_index] = 1
            y_data.append(y)
            
            # Store metadata
            block_info.append({
                'block_id': blk_id,
                'num_entries': len(group),
                'label': label,
                'vector_shape': vectors.shape
            })
        
        if not x_data:
            self.logger.error(f"No valid data processed for {mode}")
            return
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save data with compression
        output_path = Path(self.config.output_dir) / f'{mode}_data.npz'
        np.savez_compressed(
            output_path,
            x=np.array(x_data, dtype=object),
            y=np.array(y_data),
            block_info=block_info
        )
        
        # Log statistics
        block_lengths = [len(block) for block in x_data]
        self.logger.info(f"Saved {len(x_data)} {mode} blocks")
        self.logger.info(f"Block size stats - Min: {min(block_lengths)}, "
                        f"Max: {max(block_lengths)}, Avg: {np.mean(block_lengths):.1f}")
    
    def save_preprocessing_stats(self) -> None:
        """Save preprocessing statistics and metadata"""
        stats_path = Path(self.config.output_dir) / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.logger.info(f"Preprocessing statistics saved to {stats_path}")
    
    def run_preprocessing(self, structured_file: str, template_file: str, label_file: str) -> bool:
        """Main preprocessing pipeline with comprehensive error handling"""
        try:
            # Validate input files
            if not self._validate_input_files([structured_file, template_file, label_file]):
                return False
            
            # Load model
            self._load_model()
            
            # Load data files
            self.logger.info("Loading data files...")
            df_template = pd.read_csv(template_file)
            df_structured = pd.read_csv(structured_file)
            df_label = pd.read_csv(label_file)
            
            self.stats['total_logs'] = len(df_structured)
            
            self.logger.info(f"Loaded {len(df_structured)} structured logs")
            self.logger.info(f"Loaded {len(df_template)} templates")
            self.logger.info(f"Loaded {len(df_label)} labels")
            
            # Create template cache
            self.create_template_cache(df_template)
            templates = list(self.template_cache.keys())              # list of template strings
            embeddings = np.array(list(self.template_cache.values())) # array of embeddings
            
            # Ensure output directory exists
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save as .npz
            template_cache_path = os.path.join(self.config.output_dir, 'template_cache.npz')
            np.savez(template_cache_path, templates=templates, embeddings=embeddings)
            self.logger.info(f"Template cache saved to {template_cache_path}")

            
            # Convert templates to vectors
            df_structured = self.convert_templates_to_vectors(df_structured)
            
            # Extract BlockIDs
            self.logger.info("Extracting BlockIDs...")
            block_ids, extraction_stats = self.extract_blockid_from_content(df_structured['Content'])
            df_structured['BlockId'] = block_ids
            
            # Remove rows with no BlockID
            initial_count = len(df_structured)
            df_structured = df_structured.dropna(subset=['BlockId'])
            final_count = len(df_structured)
            self.stats['processed_logs'] = final_count
            self.stats['dropped_logs'] = initial_count - final_count
            
            if df_structured.empty:
                self.logger.error("No valid data remaining after BlockID extraction")
                return False
            
            self.logger.info(f"Retained {final_count}/{initial_count} rows after BlockID extraction")
            
            # Create balanced split
            extracted_blockids = set(df_structured['BlockId'].unique())
            df_label_split = self._create_balanced_split(df_label, extracted_blockids)
            
            # Merge with labels
            df_structured = pd.merge(df_structured, df_label_split, on='BlockId', how='inner')
            
            if df_structured.empty:
                self.logger.error("No data remaining after merging with labels")
                return False
            
            # Validate data consistency
            if not self._validate_data_consistency(df_structured):
                self.logger.warning("Data consistency issues detected, proceeding with caution")
            
            # Clean up unnecessary columns
            columns_to_drop = ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content', 
                             'EventId', 'EventTemplate', 'ParameterList']
            existing_columns = [col for col in columns_to_drop if col in df_structured.columns]
            df_structured.drop(columns=existing_columns, inplace=True)
            
            # Sort and prepare data
            df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
            if 'LineId' in df_structured.columns:
                df_structured.drop(columns=['LineId'], inplace=True)
            
            # Split data
            df_train = df_structured[df_structured['Usage'] == 'training'].copy()
            df_test = df_structured[df_structured['Usage'] == 'testing'].copy()
            
            # Process blocks
            self.preprocess_data_blocks(df_train, 'training')
            self.preprocess_data_blocks(df_test, 'testing')
            
            # Save statistics
            self.stats['unique_blocks'] = df_structured['BlockId'].nunique()
            self.save_preprocessing_stats()
            
            self.logger.info("Preprocessing completed successfully!")
            self.logger.info(f"Generated files in {self.config.output_dir}/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise


def main():
    """Main execution function with configuration management"""
    # Load configuration
    config = PreprocessConfig()
    
    # Override config from environment or config file if needed
    # config = load_config_from_file('config.yaml')  # Optional
    
    # Initialize preprocessor
    preprocessor = LogPreprocessor(config)
    
    # File paths
    structured_file = 'parse_result_2k_new/HDFS_2k.log_structured.csv'
    template_file = 'parse_result_2k_new/HDFS_2k.log_templates.csv'
    label_file = 'HDFS/anomaly_label.csv'
    
    # Run preprocessing
    success = preprocessor.run_preprocessing(structured_file, template_file, label_file)
    
    if success:
        print("Preprocessing completed successfully!")
        print(f"Check {config.output_dir}/ for output files")
        print("Check preprocessing.log for detailed logs")
    else:
        print("Preprocessing failed. Check preprocessing.log for details")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())