"""
Lazy DICOM Loader for efficient handling of large DICOM series

This module provides progressive loading of DICOM files to improve performance
when dealing with large datasets (>100 files).
"""

import os
import logging
import threading
import numpy as np
import pydicom
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


class LazyDICOMLoader:
    """
    Lazy loader for DICOM series that loads slices on-demand
    with intelligent caching and preloading
    """
    
    def __init__(self, dicom_files: List[str], apply_deidentification: bool = False, 
                 cache_size: int = 100, preload_window: int = 10):
        """
        Initialize the lazy DICOM loader
        
        Args:
            dicom_files: List of DICOM file paths (should be pre-sorted)
            apply_deidentification: Whether to apply face removal
            cache_size: Maximum number of slices to keep in memory
            preload_window: Number of slices to preload ahead/behind current slice
        """
        self.dicom_files = dicom_files
        self.apply_deidentification = apply_deidentification
        self.cache_size = cache_size
        self.preload_window = preload_window
        
        # Cache for loaded slices (LRU cache)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        
        # Background loading
        self._preload_thread: Optional[threading.Thread] = None
        self._stop_preload = threading.Event()
        
        # Metadata (loaded once)
        self._metadata: Optional[Dict] = None
        self._slice_shape: Optional[Tuple[int, ...]] = None
        self._dtype: Optional[np.dtype] = None
        
        # Load metadata from first file to determine shape and dtype
        self._initialize_metadata()
        
        logger.info(f"LazyDICOMLoader initialized with {len(dicom_files)} files, "
                   f"cache_size={cache_size}, preload_window={preload_window}")
    
    def _initialize_metadata(self):
        """Load metadata from the first DICOM file"""
        try:
            ds = pydicom.dcmread(self.dicom_files[0])
            
            # Get pixel array to determine shape and dtype
            pixel_array = self._process_pixel_array(ds)
            self._slice_shape = pixel_array.shape
            self._dtype = pixel_array.dtype
            
            # Extract metadata
            from utils.dicom_utils import get_dicom_metadata
            self._metadata = get_dicom_metadata(self.dicom_files[0])

            # Store last-slice IPP for precise inter-slice direction/spacing computation
            if len(self.dicom_files) > 1:
                try:
                    last_meta = get_dicom_metadata(self.dicom_files[-1])
                    if last_meta.get('ImagePositionPatient'):
                        self._metadata['ImagePositionPatientLast'] = last_meta['ImagePositionPatient']
                except Exception:
                    pass

            # Add series information
            self._metadata['SeriesInfo'] = {
                'NumberOfSlices': len(self.dicom_files),
                'SliceThickness': getattr(ds, 'SliceThickness', 0),
                'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                'PatientID': getattr(ds, 'PatientID', ''),
                'PatientName': str(getattr(ds, 'PatientName', '')),
                'StudyDate': getattr(ds, 'StudyDate', '')
            }
            
            logger.info(f"Metadata initialized: shape={self._slice_shape}, dtype={self._dtype}")
            
        except Exception as e:
            logger.error(f"Error initializing metadata: {e}")
            raise
    
    def _process_pixel_array(self, ds) -> np.ndarray:
        """Process DICOM pixel array with rescale slope and intercept"""
        pixel_array = ds.pixel_array
        
        # Apply rescale slope and intercept if available
        if hasattr(ds, 'RescaleSlope') or hasattr(ds, 'RescaleIntercept'):
            rescale_slope = getattr(ds, 'RescaleSlope', 1)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0)
            pixel_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
        
        return pixel_array
    
    def _load_slice(self, index: int) -> np.ndarray:
        """Load a single slice from disk"""
        try:
            file_path = self.dicom_files[index]
            logger.debug(f"Loading slice {index} from {file_path}")
            
            ds = pydicom.dcmread(file_path)
            pixel_array = self._process_pixel_array(ds)
            
            # Apply deidentification if requested
            if self.apply_deidentification:
                try:
                    from utils.deidentification import apply_deidentification_to_slice
                    pixel_array = apply_deidentification_to_slice(pixel_array)
                except Exception as e:
                    logger.error(f"Deidentification failed for slice {index}: {e}")
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"Error loading slice {index}: {e}")
            # Return blank slice on error
            return np.zeros(self._slice_shape, dtype=self._dtype)
    
    def get_slice(self, index: int) -> np.ndarray:
        """
        Get a slice by index (loads from cache or disk)
        
        Args:
            index: Slice index
            
        Returns:
            2D numpy array of the slice
        """
        if index < 0 or index >= len(self.dicom_files):
            raise IndexError(f"Slice index {index} out of range [0, {len(self.dicom_files)-1}]")
        
        # Check cache first
        with self._lock:
            if index in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(index)
                logger.debug(f"Slice {index} retrieved from cache")
                return self._cache[index]
        
        # Load from disk
        slice_data = self._load_slice(index)
        
        # Add to cache
        with self._lock:
            self._cache[index] = slice_data
            self._cache.move_to_end(index)
            
            # Evict old slices if cache is full
            while len(self._cache) > self.cache_size:
                evicted_idx = next(iter(self._cache))
                del self._cache[evicted_idx]
                logger.debug(f"Evicted slice {evicted_idx} from cache")
        
        # Trigger background preloading
        self._start_preload(index)
        
        return slice_data
    
    def _start_preload(self, current_index: int):
        """Start background thread to preload nearby slices"""
        # Stop any existing preload
        if self._preload_thread and self._preload_thread.is_alive():
            self._stop_preload.set()
            self._preload_thread.join(timeout=0.1)
        
        self._stop_preload.clear()
        self._preload_thread = threading.Thread(
            target=self._preload_worker,
            args=(current_index,),
            daemon=True
        )
        self._preload_thread.start()
    
    def _preload_worker(self, current_index: int):
        """Background worker to preload slices around current index"""
        # Determine range to preload
        start = max(0, current_index - self.preload_window // 2)
        end = min(len(self.dicom_files), current_index + self.preload_window // 2 + 1)
        
        logger.debug(f"Preloading slices {start} to {end-1}")
        
        for idx in range(start, end):
            if self._stop_preload.is_set():
                logger.debug("Preload interrupted")
                break
            
            # Skip if already in cache
            with self._lock:
                if idx in self._cache:
                    continue
            
            # Load slice
            try:
                slice_data = self._load_slice(idx)
                
                with self._lock:
                    # Check cache size before adding
                    if len(self._cache) >= self.cache_size:
                        # Evict oldest
                        evicted_idx = next(iter(self._cache))
                        del self._cache[evicted_idx]
                    
                    self._cache[idx] = slice_data
                    
            except Exception as e:
                logger.error(f"Error preloading slice {idx}: {e}")
    
    def get_volume(self) -> np.ndarray:
        """
        Get the full 3D volume (loads all slices)
        This may take time for large datasets
        
        Returns:
            3D numpy array
        """
        logger.info(f"Loading full volume ({len(self.dicom_files)} slices)...")
        
        slices = []
        for i in range(len(self.dicom_files)):
            slice_data = self.get_slice(i)
            slices.append(slice_data)
        
        volume = np.stack(slices)
        logger.info(f"Full volume loaded: shape={volume.shape}")
        return volume
    
    def get_metadata(self) -> Dict:
        """Get DICOM metadata"""
        return self._metadata
    
    def get_file_list(self) -> List[str]:
        """Get list of DICOM file paths"""
        return self.dicom_files
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of the volume (num_slices, height, width)"""
        return (len(self.dicom_files), *self._slice_shape)
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type of the volume"""
        return self._dtype
    
    def clear_cache(self):
        """Clear the slice cache"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown the loader and stop background threads"""
        self._stop_preload.set()
        if self._preload_thread and self._preload_thread.is_alive():
            self._preload_thread.join(timeout=1.0)
        self.clear_cache()
        logger.info("LazyDICOMLoader shutdown complete")


class LazyVolumeWrapper:
    """
    Wrapper that makes LazyDICOMLoader behave like a numpy array
    for compatibility with existing code
    """
    
    def __init__(self, loader: LazyDICOMLoader):
        self.loader = loader
        self._cached_volume: Optional[np.ndarray] = None
    
    @property
    def shape(self):
        return self.loader.shape
    
    @property
    def dtype(self):
        return self.loader.dtype
    
    def __getitem__(self, key):
        """
        Support numpy-style indexing
        For simple slice access: volume[i] returns slice i
        For full array access, loads entire volume
        """
        if isinstance(key, int):
            # Single slice access: volume[i]
            if key < 0:
                key = self.shape[0] + key
            return self.loader.get_slice(key)
        
        elif isinstance(key, slice):
            # Slice range access: volume[start:end]
            indices = range(*key.indices(self.shape[0]))
            slices = [self.loader.get_slice(i) for i in indices]
            return np.stack(slices) if slices else np.array([])
        
        elif isinstance(key, tuple):
            # Multi-dimensional indexing: volume[i, :, :] or volume[i, y, x]
            first_key = key[0]
            
            # Check if first dimension is simple integer or slice
            if isinstance(first_key, int):
                # Single slice with additional indexing: volume[i, :, :] or volume[i, y, x]
                if first_key < 0:
                    first_key = self.shape[0] + first_key
                slice_2d = self.loader.get_slice(first_key)
                
                # Apply remaining indices to the 2D slice
                if len(key) > 1:
                    remaining_key = key[1:]
                    return slice_2d[remaining_key]
                else:
                    return slice_2d
            
            elif isinstance(first_key, slice):
                # Range with additional indexing: volume[start:end, :, :]
                indices = range(*first_key.indices(self.shape[0]))
                slices = [self.loader.get_slice(i) for i in indices]
                volume_subset = np.stack(slices) if slices else np.array([])
                
                # Apply remaining indices
                if len(key) > 1:
                    remaining_key = key[1:]
                    return volume_subset[(slice(None),) + remaining_key]
                else:
                    return volume_subset
            
            else:
                # Complex indexing - need full volume
                if self._cached_volume is None:
                    logger.warning("Complex indexing requires loading full volume")
                    self._cached_volume = self.loader.get_volume()
                return self._cached_volume[key]
        
        else:
            # Complex indexing - need full volume
            if self._cached_volume is None:
                logger.warning("Complex indexing requires loading full volume")
                self._cached_volume = self.loader.get_volume()
            return self._cached_volume[key]
    
    def __array__(self):
        """Support conversion to numpy array"""
        if self._cached_volume is None:
            self._cached_volume = self.loader.get_volume()
        return self._cached_volume
    
    def astype(self, dtype):
        """Support astype conversion"""
        if self._cached_volume is None:
            self._cached_volume = self.loader.get_volume()
        return self._cached_volume.astype(dtype)
