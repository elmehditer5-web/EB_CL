"""
Optimized main processing script for PM extension decision tree
Author: Optimized from Axel Lantin's original code
Date: 2025-10-29

Key optimizations:
1. Batch processing instead of row-by-row apply
2. Parallel processing for independent operations
3. Optimized database operations with connection pooling
4. Memory-efficient data handling
"""

# =================================================================================================
# = Imports                                                                                       =
# =================================================================================================

import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Tuple

pd.options.mode.chained_assignment = None

# Import optimized functions
from utils.functions_optimized import (
    arbre_dec_optimized, 
    format_data_optimized, 
    format_push_optimized
)
from utils.db_related_functions import get_formated_data, send_to_db
from utils.variables_globales import dtype, dtype_data

# =================================================================================================
# = Optimized Main Processing                                                                    =
# =================================================================================================

def process_batch(batch_data: pd.DataFrame, data: pd.DataFrame, 
                 data_annexe: pd.DataFrame) -> pd.DataFrame:
    """Process a batch of data using vectorized operations"""
    return arbre_dec_optimized(batch_data, data, data_annexe)

def parallel_process_dataframe(data_fin: pd.DataFrame, data: pd.DataFrame, 
                              data_annexe: pd.DataFrame, n_cores: int = None) -> pd.DataFrame:
    """
    Process dataframe in parallel batches for maximum performance
    
    Args:
        data_fin: Final data to process
        data: Reference data for decision tree
        data_annexe: Annexe data
        n_cores: Number of CPU cores to use (default: auto-detect)
    
    Returns:
        Processed dataframe
    """
    if n_cores is None:
        n_cores = mp.cpu_count() - 1  # Leave one core free
    
    # If dataset is small, process without parallelization
    if len(data_fin) < 1000:
        return arbre_dec_optimized(data_fin, data, data_annexe)
    
    # Calculate optimal batch size
    batch_size = max(100, len(data_fin) // (n_cores * 4))
    
    # Split dataframe into batches
    batches = [data_fin.iloc[i:i+batch_size] for i in range(0, len(data_fin), batch_size)]
    
    print(f"Processing {len(batches)} batches on {n_cores} cores...")
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch, data, data_annexe)
            futures.append(future)
        
        # Collect results
        processed_batches = []
        for future in futures:
            processed_batches.append(future.result())
    
    # Combine results
    result = pd.concat(processed_batches, ignore_index=False)
    
    return result

def main_optimized():
    """
    Optimized main function using batch processing and vectorized operations
    
    Major optimizations:
    1. Replace .apply() with vectorized operations
    2. Use batch processing for large datasets
    3. Optimize database operations
    4. Implement parallel processing where applicable
    """
    start_time = time.time()
    
    print("=== Starting Optimized PM Extension Processing ===")
    
    # Step 1: Load and format data (already optimized in functions)
    print("Loading data from database...")
    data, data_fin, data_annexe = get_formated_data()
    print(f"Loaded {len(data_fin)} PM records for processing")
    
    # Step 2: Apply decision tree using vectorized operations
    print("Applying decision tree logic...")
    
    # Use full vectorized processing instead of row-by-row apply
    # This is the main performance improvement
    data_fin_processed = arbre_dec_optimized(data_fin, data, data_annexe)
    
    # Alternative: Use parallel processing for very large datasets
    # data_fin_processed = parallel_process_dataframe(data_fin, data, data_annexe)
    
    # Step 3: Format for push (already optimized)
    print("Formatting data for database push...")
    data_fin_formatted = format_push_optimized(data_fin_processed)
    
    # Step 4: Send to database with optimized batch inserts
    print("Sending results to database...")
    
    # Optimize database operations with chunking
    chunk_size = 5000  # Optimal chunk size for SQL Server
    for i in range(0, len(data_fin_formatted), chunk_size):
        chunk = data_fin_formatted.iloc[i:i+chunk_size]
        send_to_db(chunk, dtype, 'univers_prod_eb', 'prod_eb_kppm_zmd')
    
    # Send verification data
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        send_to_db(chunk, dtype_data, 'ODS_Prod', 'ods_prod_eb_zmd_verif')
    
    # Calculate and display performance metrics
    end_time = time.time()
    processing_time = end_time - start_time
    records_per_second = len(data_fin) / processing_time
    
    print("=== Processing Complete ===")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Records processed: {len(data_fin)}")
    print(f"Processing rate: {records_per_second:.0f} records/second")
    print(f"Performance improvement: ~{29*60/processing_time:.1f}x faster than original")
    
    return data_fin_formatted

def main():
    """
    Backward compatible main function that calls optimized version
    This maintains the same interface as the original
    """
    return main_optimized()

# =================================================================================================
# = Performance Monitoring                                                                        =
# =================================================================================================

def benchmark_performance():
    """
    Benchmark function to compare original vs optimized performance
    """
    import cProfile
    import pstats
    from io import StringIO
    
    # Profile the optimized version
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = main_optimized()
    
    profiler.disable()
    
    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 time-consuming functions
    
    print("\n=== Performance Profile ===")
    print(s.getvalue())
    
    return result

# =================================================================================================
# = Entry Point                                                                                   =
# =================================================================================================

if __name__ == '__main__':
    # Run optimized version
    main()
    
    # Uncomment to run performance benchmark
    # benchmark_performance()
