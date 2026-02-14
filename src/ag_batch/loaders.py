"""
Loaders for AlphaGenome precomputed features during training.
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional


def load_precomputed_ag_features(features_file: str) -> pd.DataFrame:
    """
    Load precomputed AlphaGenome features.
    
    Args:
        features_file: Path to the features parquet file
        
    Returns:
        DataFrame with features
    """
    try:
        df = pd.read_parquet(features_file)
        return df
    except FileNotFoundError:
        # Return empty DataFrame with expected columns if file doesn't exist
        return pd.DataFrame(columns=["variant_id", "CHROM", "POS", "REF", "ALT", "result"])


def create_variant_key(row: pd.Series) -> str:
    """
    Create a variant key for merging.
    
    Args:
        row: DataFrame row
        
    Returns:
        Variant key string
    """
    if 'variant_id' in row and pd.notna(row['variant_id']):
        return str(row['variant_id'])
    else:
        return f"{row['CHROM']}:{row['POS']}:{row['REF']}>{row['ALT']}"


def align_ag_features_with_dataset(dataset_df: pd.DataFrame, 
                                   ag_features_file: str) -> torch.Tensor:
    """
    Align AlphaGenome features with the dataset.
    
    Args:
        dataset_df: DataFrame with the dataset (including variant info)
        ag_features_file: Path to AlphaGenome features parquet file
        
    Returns:
        Tensor of aligned AlphaGenome features
    """
    # Load AlphaGenome features
    ag_df = load_precomputed_ag_features(ag_features_file)
    
    # Create variant keys for merging
    dataset_df['_variant_key'] = dataset_df.apply(create_variant_key, axis=1)
    
    if '_variant_key' not in ag_df.columns:
        ag_df['_variant_key'] = ag_df.apply(create_variant_key, axis=1)
    
    # Merge datasets
    merged_df = dataset_df.merge(ag_df, on='_variant_key', how='left')
    
    # Convert features to tensor
    # For now, we'll create zero-filled tensors as placeholders
    # In a real implementation, you would extract actual features from the 'result' column
    num_samples = len(merged_df)
    feature_dim = 1024  # Placeholder dimension, should match actual AlphaGenome feature size
    
    # Initialize with zeros
    ag_features = torch.zeros(num_samples, feature_dim, dtype=torch.float32)
    
    # Fill in non-null features (in a real implementation)
    # For now, we'll just leave them as zeros
    # In practice, you would parse the 'result' column and fill the tensor accordingly
    
    return ag_features


def add_ag_features_to_batch(batch: Dict[str, Any], 
                             ag_features_file: str) -> Dict[str, Any]:
    """
    Add AlphaGenome features to a batch.
    
    Args:
        batch: Batch dictionary from DataLoader
        ag_features_file: Path to AlphaGenome features parquet file
        
    Returns:
        Batch dictionary with AlphaGenome features added
    """
    # Create a temporary DataFrame with variant info from the batch
    batch_size = len(batch['sequence_features'])
    
    # Extract variant information from batch
    variant_info = []
    for i in range(batch_size):
        variant_data = {}
        if 'variant_id' in batch:
            variant_data['variant_id'] = batch['variant_id'][i]
        if 'chromosome' in batch:
            variant_data['CHROM'] = batch['chromosome'][i]
        if 'position' in batch:
            variant_data['POS'] = batch['position'][i]
        if 'ref' in batch:
            variant_data['REF'] = batch['ref'][i]
        if 'alt' in batch:
            variant_data['ALT'] = batch['alt'][i]
        variant_info.append(variant_data)
    
    batch_df = pd.DataFrame(variant_info)
    
    # Align AlphaGenome features
    ag_features = align_ag_features_with_dataset(batch_df, ag_features_file)
    
    # Add to batch
    batch['ag_feats'] = ag_features
    
    return batch