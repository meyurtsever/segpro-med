# KoGa: Medical Image Annotation

## Software Overview

KoGa is a comprehensive software solution developed for the annotation and analysis of medical imaging data. It is structured into several distinct modules, each accessible through a tab-based interface. These modules facilitate a range of tasks from data visualization and manual/AI-assisted annotation to format conversion and collaborative project management. The software provides support for multiple medical imaging modalities including CT and MRI scans, with specialized support for brain imaging studies. Advanced features include an intelligent retrieval system for patient data, automated data anonymization capabilities, and integration with multiple Vision-Language Models for enhanced analysis.

The primary sections of the software include the Viewer, Editor, Conversion, Label Manager, and Crowdsourcing modules. Each module is designed to address specific workflows in medical image annotation and analysis, from basic viewing and navigation to advanced AI-assisted segmentation and collaborative annotation projects.

KoGa distinguishes itself from existing annotation tools through its comprehensive feature set. Unlike traditional solutions, it combines web-based accessibility with advanced AI integration, supporting both manual annotation workflows and automated segmentation through Vision-Language Models. The software provides multi-modality support for different imaging types, incorporates a sophisticated retrieval system for patient data management, and maintains an open-source architecture. A detailed comparison of these features with existing tools is presented in the comparative analysis table.

## Image Retrieval - Search

The Image Retrieval system is designed to provide efficient access to medical imaging datasets through an intelligent search mechanism. The system operates on structured medical imaging directories and maintains a cached index of patient data organized by anomaly classes and patient identifiers.

The search functionality enables users to query patient data using partial matching algorithms that search across patient identifiers, anomaly classifications, and folder structures. The retrieval system automatically scans the root directory structure to identify and catalog available patient datasets, organizing them according to predefined hierarchical patterns. When a search query is submitted, the system returns ranked results prioritizing exact matches in anomaly classes and patient folders, followed by partial matches based on query position within the search text.

The system supports real-time search with dropdown-based selection mechanisms, allowing users to quickly locate and load specific patient datasets. Upon selection, the system automatically retrieves associated FLAIR imaging data and corresponding segmentation files when available, providing immediate access to both raw imaging data and any existing annotations.

## Data Anonymization

Medical imaging data contains sensitive patient information that requires careful handling to ensure privacy compliance. DICOM files typically include metadata fields such as patient names, birth dates, patient identifiers, sex, study dates, and other personally identifiable information that must be removed before data sharing or analysis.

KoGa implements automated data anonymization processes that are applied during data loading and export operations. The anonymization pipeline utilizes pydeface technology for facial structure removal in brain imaging studies, preventing patient identification through facial features visible in MRI and CT scans. The software automatically detects and processes sensitive metadata fields within DICOM headers, removing or replacing identifiable information while preserving clinically relevant technical parameters.

The anonymization process is applied transparently during data import, ensuring that sensitive information is cleansed before the data enters the annotation workflow. When exporting annotated data, the system maintains the anonymized state, providing secure data sharing capabilities for research and collaborative purposes. The process preserves essential imaging parameters such as voxel spacing, image orientation, and sequence information required for proper medical analysis while removing patient-specific identifiers.

## Multiple Medical Data Support

KoGa provides comprehensive support for multiple medical imaging modalities and anatomical regions. The software is designed to handle various imaging types including Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) studies across different anatomical regions with specialized optimization for brain imaging applications.

The system supports different MRI sequences including T1-weighted, T2-weighted, FLAIR (Fluid Attenuated Inversion Recovery), and other specialized sequences commonly used in clinical practice. Each sequence type is processed with appropriate windowing parameters extracted from DICOM metadata to ensure optimal visualization and analysis. The software automatically detects sequence characteristics and applies suitable preprocessing workflows.

For brain imaging specifically, the system includes specialized tools for anatomical structure detection and region-of-interest identification. The brain-specific modules support multi-planar reconstruction capabilities, enabling viewing in axial, sagittal, and coronal orientations. Advanced features include automatic detection of brain orientation patterns and anatomical landmark identification to assist in standardized annotation workflows.

The multi-modality support extends to different organs and body regions, with the system capable of processing imaging studies from various anatomical locations. The flexible architecture allows for the integration of organ-specific processing modules while maintaining consistent user interfaces and annotation workflows across different imaging types.

## Viewer

The Viewer tab is designed for the inspection of pre-annotated medical imaging datasets. A core feature of this module is the integrated retrieval system that allows users to query and load data from supported datasets through intelligent search mechanisms. The search functionality operates on structured patient directories, enabling rapid location of specific cases through partial matching algorithms that search across patient identifiers and anomaly classifications.

When a dataset is loaded, the system displays the medical images along with any available segmentation overlays and corresponding labels. The interface automatically applies appropriate windowing parameters extracted from DICOM metadata to ensure optimal image contrast and visualization quality.

Navigation through different slices of the medical scan is facilitated by both a slider interface and dedicated control buttons, supporting efficient traversal through large volumetric datasets. The interface also presents relevant metadata associated with the imaging data, including technical parameters, patient demographics (when not anonymized), and study information.

Users are provided with comprehensive export options for the visualized data. This can be performed for the entire series of slices or for individual, specific slices. Multiple image formats are supported for export, including standard formats such as PNG for general visualization and specialized medical formats for clinical use.

## Editor

The Editor tab provides a comprehensive suite of tools for the annotation of medical images across multiple modalities including CT and MRI studies. Users can load either individual image slices or complete medical scan volumes into the editor environment. The annotation process can be performed through manual drawing tools or with the assistance of advanced artificial intelligence models.

For manual annotation, a custom plugin developed for the Gradio framework is integrated, which supports the creation of various geometric shapes including rectangles, circles, polygons, and freehand forms. These tools provide precise control over annotation boundaries and support complex anatomical structure delineation.

For AI-assisted annotation, the software incorporates multiple state-of-the-art models. The SAM2 (Segment Anything Model 2) and MedSAM models are available for sophisticated segmentation tasks, where annotations can be generated based on user-defined points of interest or through automated prompt-less mechanisms. Following machine-generated annotation, users have the option to review, modify, accept, or discard the suggested segmentations, maintaining full control over the annotation quality.

The Editor also integrates three specialized Vision-Language Models (VLMs) for advanced medical data analysis: MedGemma-4B, Med-R1, and SmolVLM. Users can interact with these models using two pre-defined medical imaging prompts or by creating custom queries tailored to specific analysis requirements. A notable feature is the voice-to-text capability, which enables users to generate prompts through voice input that is transcribed into text using Whisper models. This facilitates hands-free operation and rapid query generation during annotation workflows.

Furthermore, the VLMs can be utilized to generate candidate labels for patient data being analyzed, providing automated suggestions for anatomical structures and pathological findings. This feature enhances annotation efficiency by providing expert-level suggestions that can be validated and refined by human annotators.

## Conversion

The Conversion tab offers comprehensive functionality for converting medical image files between different standardized formats. This utility is essential for ensuring compatibility across various imaging systems, analysis pipelines, and research workflows. The conversion engine supports bidirectional transformation between several widely-used medical imaging formats.

The software supports conversion operations between DICOM (Digital Imaging and Communications in Medicine), NIFTI (Neuroimaging Informatics Technology Initiative), MAT (MATLAB data format), and PNG (Portable Network Graphics) formats. These conversions maintain essential metadata and imaging parameters while adapting the data structure to the requirements of different software platforms and analysis tools.

The conversion process includes automatic preservation of critical imaging parameters such as voxel spacing, image orientation, and windowing information. For DICOM to NIFTI conversions, the system maintains spatial coordinate systems and slice ordering information. The interface provides options for batch processing of multiple files and directory-based conversion workflows to support large-scale data processing requirements.

## Label Manager

    The Label Manager module is designed to streamline the annotation workflow by allowing users to manage sets of labels. It supports the import of label sets from files compatible with the ITK-SNAP software. Once imported, these label sets can be edited and saved within the KoGa environment. This feature enables the quick application of pre-defined, standardized labels during the annotation process, enhancing consistency and efficiency.

## Crowdsourcing

The Crowdsourcing module facilitates collaborative annotation projects through a role-based system involving administrators and expert annotators. An administrator can initiate an annotation campaign by defining a set of tasks, which consist of patient data requiring annotation.

These tasks are then assigned to expert users. Experts can view their assigned tasks in a dedicated panel, perform the required annotations using the Editor tools, and submit their work back to the administrator. The system allows the administrator to monitor the progress of the campaign, review the submitted annotations, and provide feedback by rating the quality of the work. Based on this review, the administrator can formally accept or reject the annotations, completing the workflow.
