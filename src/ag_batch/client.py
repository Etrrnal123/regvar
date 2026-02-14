"""
AlphaGenome Batch Scorer with concurrency, rate limiting, caching and resume capabilities.
"""
import os
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Iterable
import pandas as pd
import numpy as np

# 直接导入 AlphaGenome 模块，缺少时应直接报错
try:
    from alphagenome.models import dna_client
    from alphagenome.data import genome
except ImportError as e:
    raise ImportError(f"无法导入 AlphaGenome 模块: {e}. 请确保已正确安装 AlphaGenome 库。")

import tqdm


class AlphaGenomeBatchScorer:
    def __init__(self, api_key: str, organism: str = "HOMO_SAPIENS", 
                 seq_len: int = 131072, scorers: List[str] = None,
                 cache_dir: str = "ag_cache", max_workers: int = 12, 
                 rate_limit: float = 8.0, max_retries: int = 5):
        """
        Initialize the AlphaGenome batch scorer.
        
        Args:
            api_key: API key for AlphaGenome
            organism: Organism name
            seq_len: Sequence length for predictions
            scorers: List of scorers to use
            cache_dir: Directory for caching results
            max_workers: Maximum number of concurrent workers
            rate_limit: Maximum requests per second
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.organism = organism
        self.seq_len = seq_len
        self.scorers = scorers or ["RNA_SEQ"]
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the AlphaGenome client
        self.client = dna_client.create(api_key)
        
        # Rate limiting variables
        self._last_request_time = 0
        self._lock = threading.Lock()

    def _cache_key(self, row: Dict[str, Any]) -> str:
        """
        Generate a cache key for a given variant.
        """
        # Handle both 'chr' and 'CHROM' column names
        chrom = row.get('CHROM') or row.get('chr', '')
        pos = row.get('POS') or row.get('pos', '')
        ref = row.get('REF') or row.get('ref', '')
        alt = row.get('ALT') or row.get('alt', '')
        
        key_string = f"{chrom}:{pos}:{ref}>{alt}:" \
                     f"{self.seq_len}:{self.organism}:{sorted(self.scorers)}"
        return hashlib.sha1(key_string.encode()).hexdigest()

    def _build_cache_index(self) -> set:
        """
        Build an index of existing cache files to speed up cache checks.
        """
        cache_index = set()
        if os.path.exists(self.cache_dir):
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    # 提取缓存键（去掉.json扩展名）
                    cache_key = file_name[:-5]  # 移除 '.json'
                    cache_index.add(cache_key)
        return cache_index

    def _from_cache(self, key: str) -> Dict[str, Any]:
        """
        Retrieve result from cache if it exists.
        """
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _to_cache(self, key: str, payload: Dict[str, Any]) -> None:
        """
        Save result to cache.
        """
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(payload, f)

    def _score_one(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single variant with retry and rate limiting.
        """
        cache_key = self._cache_key(row)
        
        # Try to get from cache first
        cached_result = self._from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Rate limiting
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < 1.0 / self.rate_limit:
                time.sleep(1.0 / self.rate_limit - elapsed)
            self._last_request_time = time.time()
        
        # Handle column name variations
        chrom = row.get('CHROM') or row.get('chr', '')
        pos = row.get('POS') or row.get('pos', '')
        ref = row.get('REF') or row.get('ref', '')
        alt = row.get('ALT') or row.get('alt', '')
        variant_id = row.get("variant_id", f"{chrom}:{pos}:{ref}>{alt}")
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Create interval and variant
                flank_size = self.seq_len // 2
                interval = genome.Interval(
                    chromosome=f"chr{chrom}",
                    start=max(0, int(pos) - flank_size),
                    end=int(pos) + flank_size
                )
                
                variant = genome.Variant(
                    chromosome=f"chr{chrom}",
                    position=int(pos),
                    reference_bases=ref,
                    alternate_bases=alt
                )
                
                # Score the variant
                # Check if predict_variants method exists
                if hasattr(self.client, 'predict_variants'):
                    # Convert string scorers to OutputType enums
                    output_types = []
                    for scorer in self.scorers:
                        if hasattr(dna_client.OutputType, scorer):
                            output_types.append(getattr(dna_client.OutputType, scorer))
                        else:
                            print(f"Warning: Unknown scorer {scorer}")
                    
                    result = self.client.predict_variants(
                        intervals=[interval],
                        variants=[variant],
                        requested_outputs=output_types,
                        ontology_terms=None  # 必须显式传入 ontology_terms=None
                    )
                else:
                    # Fallback to score_variant if predict_variants doesn't exist
                    result = self.client.score_variant(
                        interval=interval,
                        variant=variant,
                        organism=getattr(dna_client.Organism, self.organism),
                    )
                
                # Convert result to serializable format
                result_dict = {
                    "variant_id": variant_id,
                    "CHROM": chrom,
                    "POS": pos,
                    "REF": ref,
                    "ALT": alt,
                    "result": str(result)  # Simplified for now
                }
                
                # Cache the result
                self._to_cache(cache_key, result_dict)
                return result_dict
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt, return failure
                    return {
                        "variant_id": variant_id,
                        "CHROM": chrom,
                        "POS": pos,
                        "REF": ref,
                        "ALT": alt,
                        "error": str(e)
                    }
                else:
                    # Exponential backoff with jitter
                    sleep_time = (2 ** attempt) + np.random.uniform(0, 1)
                    time.sleep(sleep_time)
        
        # Should not reach here
        return None

    def score_many(self, rows_iterable: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple variants concurrently with rate limiting and caching.
        
        Args:
            rows_iterable: Iterable of variant rows
            
        Returns:
            List of results
        """
        rows = list(rows_iterable)
        results = [None] * len(rows)
        
        # 预加载缓存索引以提高检查速度
        print("Building cache index...")
        cache_index = self._build_cache_index()
        cached_count = 0
        
        # 预先检查缓存，统计已缓存的变体数量
        for i, row in enumerate(rows):
            cache_key = self._cache_key(row)
            if cache_key in cache_index:
                cached_result = self._from_cache(cache_key)
                if cached_result is not None:
                    results[i] = cached_result
                    cached_count += 1
        
        print(f"Found {cached_count} cached results out of {len(rows)} total variants")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit only uncached tasks
            future_to_index = {}
            uncached_indices = [i for i, r in enumerate(results) if r is None]
            
            print(f"Submitting {len(uncached_indices)} variants for processing...")
            
            for i in uncached_indices:
                row = rows[i]
                future_to_index[executor.submit(self._score_one, row)] = i
            
            # Collect results with progress bar
            if future_to_index:  # 只有当有待处理的任务时才显示进度条
                for future in tqdm.tqdm(as_completed(future_to_index), 
                                      total=len(future_to_index),
                                      desc="Scoring variants"):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        # Handle any unexpected exceptions
                        row = rows[index]
                        chrom = row.get('CHROM') or row.get('chr', '')
                        pos = row.get('POS') or row.get('pos', '')
                        ref = row.get('REF') or row.get('ref', '')
                        alt = row.get('ALT') or row.get('alt', '')
                        variant_id = row.get("variant_id", f"{chrom}:{pos}:{ref}>{alt}")
                        
                        results[index] = {
                            "variant_id": variant_id,
                            "CHROM": chrom,
                            "POS": pos,
                            "REF": ref,
                            "ALT": alt,
                            "error": str(e)
                        }
            else:
                print("All variants are already cached. No processing needed.")
        
        return results
