"""
Patient Retrieval System

This module provides functionality to search and retrieve patient data
from the structured medical imaging directory.
"""

import os
import glob
from typing import List, Tuple, Dict, Optional
from utils.debug_utils import logger

class PatientRetrieval:
    """Handle patient data retrieval from the medical imaging directory structure"""
    
    def __init__(self, root_directory: str = r"C:\Gazi\TR_TBP_Anonymised_enc\Anonymised\500 MR"):
        """
        Initialize the patient retrieval system
        
        Args:
            root_directory: Root directory containing anomaly class folders
        """
        self.root_directory = root_directory
        self._patient_cache = {}
        self._cache_initialized = False
    
    def _initialize_patient_cache(self) -> None:
        """Initialize the patient cache by scanning the directory structure"""
        if self._cache_initialized:
            return
            
        logger.info(f"Initializing patient cache from: {self.root_directory}")
        self._patient_cache = {}
        
        if not os.path.exists(self.root_directory):
            logger.warning(f"Root directory does not exist: {self.root_directory}")
            self._cache_initialized = True
            return
        
        try:
            # Scan anomaly class directories (e.g., HGG, LGG, etc.)
            for anomaly_class in os.listdir(self.root_directory):
                anomaly_path = os.path.join(self.root_directory, anomaly_class)
                
                if not os.path.isdir(anomaly_path):
                    continue
                
                # Scan patient folders within each anomaly class
                for patient_folder in os.listdir(anomaly_path):
                    patient_path = os.path.join(anomaly_path, patient_folder)
                    
                    if not os.path.isdir(patient_path):
                        continue
                    
                    # Look for FLAIR or flair folders
                    flair_path = self._find_flair_folder(patient_path)
                    
                    if flair_path:
                        # Store patient info
                        display_name = f"{anomaly_class}/{patient_folder}"
                        self._patient_cache[display_name] = {
                            'anomaly_class': anomaly_class,
                            'patient_folder': patient_folder,
                            'flair_path': flair_path,
                            'segmentation_path': self._find_segmentation_file(flair_path)
                        }
            
            logger.info(f"Patient cache initialized with {len(self._patient_cache)} entries")
            self._cache_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing patient cache: {e}")
            self._cache_initialized = True
    
    def _find_flair_folder(self, patient_path: str) -> Optional[str]:
        """
        Find FLAIR or flair folder within a patient directory
        
        Args:
            patient_path: Path to the patient directory
            
        Returns:
            Path to the FLAIR folder containing DICOM files, or None if not found
        """
        # Look for FLAIR or flair folders (case insensitive)
        for folder_name in os.listdir(patient_path):
            if folder_name.lower() in ['flair', 'flair.']:
                flair_candidate = os.path.join(patient_path, folder_name)
                
                if not os.path.isdir(flair_candidate):
                    continue
                
                # Check if this folder contains DICOM files
                if self._has_dicom_files(flair_candidate):
                    return flair_candidate
                
                # If no DICOM files, check one level deeper
                for subfolder in os.listdir(flair_candidate):
                    subfolder_path = os.path.join(flair_candidate, subfolder)
                    if os.path.isdir(subfolder_path) and self._has_dicom_files(subfolder_path):
                        return subfolder_path
        
        return None
    
    def _has_dicom_files(self, directory: str) -> bool:
        """
        Check if a directory contains DICOM files
        
        Args:
            directory: Directory path to check
            
        Returns:
            True if directory contains .dcm files
        """
        try:
            dicom_files = glob.glob(os.path.join(directory, "*.dcm"))
            return len(dicom_files) > 0
        except Exception:
            return False
    
    def _find_segmentation_file(self, flair_path: str) -> Optional[str]:
        """
        Find the segmentation file (Untitled.nii) in the FLAIR directory
        
        Args:
            flair_path: Path to the FLAIR directory
            
        Returns:
            Path to the segmentation file, or None if not found
        """
        if not flair_path:
            return None
            
        # Look for Untitled.nii in the flair directory
        flair_dir = os.path.dirname(flair_path) if os.path.isfile(flair_path) else flair_path
        segmentation_file = os.path.join(flair_dir, "Untitled.nii")
        
        if os.path.exists(segmentation_file):
            return segmentation_file
        
        # Also check for .nii.gz version
        segmentation_file_gz = os.path.join(flair_dir, "Untitled.nii.gz")
        if os.path.exists(segmentation_file_gz):
            return segmentation_file_gz
        
        return None
    
    def search_patients(self, query: str) -> List[str]:
        """
        Search for patients matching the query string
        
        Args:
            query: Search query string
            
        Returns:
            List of matching patient display names
        """
        self._initialize_patient_cache()
        
        if not query.strip():
            return []
        
        query_lower = query.lower()
        matches = []
        
        for display_name, patient_info in self._patient_cache.items():
            # Search in display name, anomaly class, and patient folder
            search_text = f"{display_name} {patient_info['anomaly_class']} {patient_info['patient_folder']}".lower()
            
            if query_lower in search_text:
                matches.append(display_name)
        
        # Sort matches to prioritize exact matches and closer matches
        matches.sort(key=lambda x: (
            query_lower not in x.lower().split('/')[0],  # Anomaly class exact match first
            query_lower not in x.lower().split('/')[-1],  # Patient folder exact match second
            x.lower().find(query_lower),  # Then by position of match
            x  # Finally alphabetically
        ))
        
        return matches[:50]  # Limit to 50 results for performance
    
    def get_patient_info(self, display_name: str) -> Optional[Dict]:
        """
        Get detailed information for a specific patient
        
        Args:
            display_name: Patient display name (e.g., "HGG/hgg (4)")
            
        Returns:
            Dictionary containing patient information or None if not found
        """
        self._initialize_patient_cache()
        return self._patient_cache.get(display_name)
    
    def get_all_patients(self) -> List[str]:
        """
        Get all available patient display names
        
        Returns:
            List of all patient display names
        """
        self._initialize_patient_cache()
        return list(self._patient_cache.keys())
    
    def is_directory_valid(self) -> bool:
        """
        Check if the root directory exists and is accessible
        
        Returns:
            True if directory exists and is accessible
        """
        return os.path.exists(self.root_directory) and os.path.isdir(self.root_directory)
    
    def get_directory_stats(self) -> Dict:
        """
        Get statistics about the directory structure
        
        Returns:
            Dictionary with directory statistics
        """
        self._initialize_patient_cache()
        stats = {
            'root_directory': self.root_directory,
            'directory_exists': self.is_directory_valid(),
            'total_patients': len(self._patient_cache),
            'anomaly_classes': {}
        }
        
        for display_name, patient_info in self._patient_cache.items():
            anomaly_class = patient_info['anomaly_class']
            if anomaly_class not in stats['anomaly_classes']:
                stats['anomaly_classes'][anomaly_class] = 0
            stats['anomaly_classes'][anomaly_class] += 1
        
        return stats

# Global instance for easy access
patient_retrieval = PatientRetrieval()

def set_root_directory(root_dir: str) -> None:
    """
    Set a custom root directory for patient retrieval
    
    Args:
        root_dir: Path to the root directory containing patient data
    """
    global patient_retrieval
    patient_retrieval.root_directory = root_dir
    patient_retrieval._cache_initialized = False  # Force re-initialization
    patient_retrieval._patient_cache = {}
