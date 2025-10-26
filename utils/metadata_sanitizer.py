"""
DICOM Metadata Sanitizer

This module provides utilities to sanitize patient-related metadata
before displaying in the UI. It ensures PHI (Protected Health Information)
is anonymized for display purposes only, without altering the source files.

Based on DICOM Patient Module specifications:
https://dicom.innolitics.com/ciods/computed-radiography-image/patient
"""

import logging

logger = logging.getLogger(__name__)

# Patient Module Tags from DICOM Standard
# Reference: https://dicom.innolitics.com/ciods/computed-radiography-image/patient
PATIENT_IDENTIFIABLE_FIELDS = [
    # Primary patient identifiers
    'PatientName',              # (0010,0010)
    'PatientID',                # (0010,0020)
    
    # Birth and demographic information
    'PatientBirthDate',         # (0010,0030)
    'PatientBirthTime',         # (0010,0032)
    'PatientBirthDateInAlternativeCalendar',  # (0010,0033)
    
    # Physical characteristics
    'PatientSex',               # (0010,0040) - Keep this as it's clinical
    'PatientAge',               # (0010,1010) - Keep this as it's clinical
    'PatientSize',              # (0010,1020) - Keep for clinical context
    'PatientWeight',            # (0010,1030) - Keep for clinical context
    'PatientAddress',           # (0010,1040)
    
    # Contact information
    'PatientTelephoneNumbers',  # (0010,2154)
    'PatientTelecomInformation',# (0010,2155)
    
    # Other identifiers
    'OtherPatientIDs',          # (0010,1000)
    'OtherPatientNames',        # (0010,1001)
    'OtherPatientIDsSequence',  # (0010,1002)
    
    # Ethnic and religious information
    'EthnicGroup',              # (0010,2160)
    'PatientReligiousPreference',# (0010,21F0)
    
    # Comments that might contain PHI
    'PatientComments',          # (0010,4000)
    
    # Insurance and billing (if present)
    'PatientInsurancePlanCodeSequence',  # (0010,0050)
    'PatientPrimaryLanguageCodeSequence', # (0010,0101)
    'PatientPrimaryLanguageModifierCodeSequence', # (0010,0102)
    
    # Additional patient history
    'AdmittingDiagnosesDescription',     # (0008,1080)
    'AdmittingDiagnosesCodeSequence',    # (0008,1084)
    'PatientState',                       # (0038,0500)
    'PatientClinicalTrialParticipationSequence', # (0012,0030)
    
    # Responsible person information (also PHI)
    'ResponsiblePerson',                  # (0010,2297)
    'ResponsiblePersonRole',              # (0010,2298)
    'ResponsibleOrganization',            # (0010,2299)
    
    # Additional identifiers
    'IssuerOfPatientID',                  # (0010,0021)
    'IssuerOfPatientIDQualifiersSequence', # (0010,0024)
    'TypeOfPatientID',                    # (0010,0022)
    
    # Military rank and occupation (personal identifiers)
    'MilitaryRank',                       # (0010,1080)
    'Occupation',                         # (0010,2180)
    
    # Pregnancy status fields (keep for clinical value but mark as clinical, not PHI)
    # 'PatientSexNeutered',               # (0010,2203) - Keep for veterinary
    # 'PregnancyStatus',                  # (0010,21C0) - Keep for clinical context
    # 'LastMenstrualDate',                # (0010,21D0) - Keep for clinical context
]

# Fields to KEEP for clinical relevance (not PHI but useful for interpretation)
CLINICAL_FIELDS_TO_KEEP = [
    'PatientSex',               # Clinical relevance
    'PatientAge',               # Clinical relevance
    'PatientSize',              # Clinical relevance
    'PatientWeight',            # Clinical relevance
    'PregnancyStatus',          # Clinical relevance
    'LastMenstrualDate',        # Clinical relevance
    'PatientSexNeutered',       # For veterinary use
    'Modality',                 # Not PHI
    'BodyPartExamined',         # Not PHI
]


def sanitize_metadata_for_display(metadata):
    """
    Sanitize patient metadata for display in the UI.
    
    Replaces patient identifiable information with empty strings
    while preserving clinical and technical metadata.
    
    Args:
        metadata (dict): Original metadata dictionary from DICOM/NIfTI
        
    Returns:
        dict: Sanitized metadata with PHI fields replaced with ""
        
    Note:
        This function creates a COPY of the metadata and does NOT
        modify the original dictionary or the source files.
    """
    if not metadata:
        return {}
    
    # Create a deep copy to avoid modifying the original
    import copy
    sanitized = copy.deepcopy(metadata)
    
    # Track which fields were sanitized for logging
    sanitized_fields = []
    
    # Sanitize top-level fields
    for field in PATIENT_IDENTIFIABLE_FIELDS:
        if field in sanitized and field not in CLINICAL_FIELDS_TO_KEEP:
            # Replace with empty string for display
            original_value = sanitized[field]
            sanitized[field] = ""
            sanitized_fields.append(field)
            logger.debug(f"Sanitized field '{field}': '{original_value}' -> ''")
    
    # Handle nested dictionaries (like SeriesInfo)
    if 'SeriesInfo' in sanitized and isinstance(sanitized['SeriesInfo'], dict):
        series_info = sanitized['SeriesInfo']
        for field in PATIENT_IDENTIFIABLE_FIELDS:
            if field in series_info and field not in CLINICAL_FIELDS_TO_KEEP:
                original_value = series_info[field]
                series_info[field] = ""
                sanitized_fields.append(f"SeriesInfo.{field}")
                logger.debug(f"Sanitized nested field 'SeriesInfo.{field}': '{original_value}' -> ''")
    
    # Handle any other nested structures that might contain patient info
    for key, value in sanitized.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_nested_dict(value)
    
    if sanitized_fields:
        logger.info(f"Sanitized {len(sanitized_fields)} PHI fields for display: {', '.join(sanitized_fields)}")
    else:
        logger.debug("No PHI fields found in metadata")
    
    return sanitized


def sanitize_nested_dict(data):
    """
    Recursively sanitize nested dictionaries.
    
    Args:
        data (dict): Dictionary that may contain PHI
        
    Returns:
        dict: Sanitized copy of the dictionary
    """
    if not isinstance(data, dict):
        return data
    
    import copy
    sanitized = copy.deepcopy(data)
    
    for field in PATIENT_IDENTIFIABLE_FIELDS:
        if field in sanitized and field not in CLINICAL_FIELDS_TO_KEEP:
            sanitized[field] = ""
    
    # Recursively process nested dictionaries
    for key, value in sanitized.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_nested_dict(value)
    
    return sanitized


def get_phi_field_names():
    """
    Get list of PHI field names that should be sanitized.
    
    Returns:
        list: Field names that contain PHI
    """
    return [field for field in PATIENT_IDENTIFIABLE_FIELDS 
            if field not in CLINICAL_FIELDS_TO_KEEP]


def is_phi_field(field_name):
    """
    Check if a field name contains PHI.
    
    Args:
        field_name (str): Name of the field to check
        
    Returns:
        bool: True if field contains PHI, False otherwise
    """
    return (field_name in PATIENT_IDENTIFIABLE_FIELDS and 
            field_name not in CLINICAL_FIELDS_TO_KEEP)


def sanitize_for_logging(value, field_name):
    """
    Sanitize a value for logging purposes.
    
    Args:
        value: Original value
        field_name (str): Name of the field
        
    Returns:
        str: Sanitized value safe for logging
    """
    if is_phi_field(field_name):
        # For PHI fields, return a masked representation
        if not value or value == "":
            return "[EMPTY]"
        return "[REDACTED]"
    return str(value)


def create_display_metadata_summary(metadata):
    """
    Create a human-readable summary of metadata for display.
    
    Args:
        metadata (dict): Sanitized metadata
        
    Returns:
        str: Formatted summary string
    """
    if not metadata:
        return "No metadata available"
    
    summary_lines = []
    
    # Add modality if available
    if 'Modality' in metadata:
        summary_lines.append(f"Modality: {metadata['Modality']}")
    
    # Add dimensions if available
    if 'Rows' in metadata and 'Columns' in metadata:
        summary_lines.append(f"Dimensions: {metadata['Columns']} × {metadata['Rows']}")
    
    # Add spacing if available
    if 'PixelSpacing' in metadata:
        spacing = metadata['PixelSpacing']
        if isinstance(spacing, (list, tuple)) and len(spacing) >= 2:
            summary_lines.append(f"Pixel Spacing: {spacing[0]:.2f} × {spacing[1]:.2f} mm")
    
    # Add slice thickness if available
    if 'SliceThickness' in metadata:
        summary_lines.append(f"Slice Thickness: {metadata['SliceThickness']} mm")
    
    # Add series description if available
    if 'SeriesDescription' in metadata:
        summary_lines.append(f"Series: {metadata['SeriesDescription']}")
    
    # Add study date if available (not PHI)
    if 'StudyDate' in metadata:
        summary_lines.append(f"Study Date: {metadata['StudyDate']}")
    
    # Note about sanitization
    summary_lines.append("\n[Patient information has been anonymized for display]")
    
    return "\n".join(summary_lines)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample metadata
    sample_metadata = {
        'PatientName': 'John Doe',
        'PatientID': '12345678',
        'PatientBirthDate': '19800101',
        'PatientSex': 'M',
        'PatientAge': '043Y',
        'Modality': 'MR',
        'SeriesDescription': 'T1 MPRAGE',
        'Rows': 512,
        'Columns': 512,
        'PixelSpacing': [0.5, 0.5],
        'SliceThickness': 1.0,
        'SeriesInfo': {
            'PatientID': '12345678',
            'PatientName': 'John Doe',
            'NumberOfSlices': 176
        }
    }
    
    print("Original metadata:")
    print(sample_metadata)
    print("\n" + "="*50 + "\n")
    
    sanitized = sanitize_metadata_for_display(sample_metadata)
    print("Sanitized metadata:")
    print(sanitized)
    print("\n" + "="*50 + "\n")
    
    print("Display summary:")
    print(create_display_metadata_summary(sanitized))
