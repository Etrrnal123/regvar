"""
Load DNA features and labels with genomic variant information.
"""
import pandas as pd
import numpy as np
import os

def load_dna_fea_and_labels(fea_dir):
    """
    加载DNA特征、标签以及基因组变异信息

    Args:
        fea_dir (str): 特征文件所在目录

    Returns:
        tuple: (features, labels, chromosomes, positions, refs, alts)
    """
    # 加载序列特征和标签
    features = np.load(os.path.join(fea_dir, "features.npy"))
    labels = np.load(os.path.join(fea_dir, "labels.npy"))
    
    # 初始化基因组变异信息为None
    chromosomes, positions, refs, alts = None, None, None, None
    
    # 如果存在基因组数据文件，则加载
    genomic_data_path = os.path.join(fea_dir, "genomic_data.csv")
    if os.path.exists(genomic_data_path):
        df = pd.read_csv(genomic_data_path)
        
        # 提取基因组变异信息（处理列名的大小写变体）
        chromosomes = df.get('chr', df.get('CHROM', None)).values
        positions = df.get('pos', df.get('POS', None)).values
        refs = df.get('ref', df.get('REF', None)).values
        alts = df.get('alt', df.get('ALT', None)).values
        
        # 确保标签一致
        if 'label' in df.columns:
            genomic_labels = df['label'].values
            # 检查标签是否一致
            if not np.array_equal(labels, genomic_labels):
                print("Warning: Labels in genomic data do not match labels in features. Using feature labels.")

    return features, labels, chromosomes, positions, refs, alts