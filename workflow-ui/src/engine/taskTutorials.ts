export interface TaskTutorialStep {
  nodeKey: string;
  title: string;
  body: string;
  placement?: 'right' | 'left' | 'top' | 'bottom';
}

export interface TaskTutorialConfig {
  templateId: string;
  title: string;
  storageKey: string;
  steps: TaskTutorialStep[];
}

export const PHI_DEIDENTIFICATION_TUTORIAL: TaskTutorialConfig = {
  templateId: 'phi-deidentification',
  title: 'PHI Deidentification Walkthrough',
  storageKey: 'segpro-med.workflow.phiDeidentificationTutorialDismissed',
  steps: [
    {
      nodeKey: 'loader',
      title: 'Load the source study',
      body: 'Start here. Data Loader accepts a DICOM directory, uploaded ZIP, single DICOM file, or NIfTI volume and creates the session used by the rest of the workflow.',
      placement: 'right',
    },
    {
      nodeKey: 'metadataBefore',
      title: 'Inspect metadata before sanitization',
      body: 'This viewer shows the metadata as loaded. Use it to confirm which patient, study, institution, and acquisition fields exist before deidentification runs.',
      placement: 'bottom',
    },
    {
      nodeKey: 'deidentify',
      title: 'Create the sanitized copy',
      body: 'The Deidentify node writes a new output. It blanks DICOM PHI fields, removes private tags, clears supported NIfTI text headers, and logs the operation to the audit trail.',
      placement: 'right',
    },
    {
      nodeKey: 'metadataAfter',
      title: 'Verify the sanitized metadata',
      body: 'After the workflow runs, this viewer should still show useful technical metadata while PHI fields are empty or removed. This is the verification point.',
      placement: 'bottom',
    },
    {
      nodeKey: 'export',
      title: 'Use the sanitized output',
      body: 'Export Sanitized Data shows the output path that can be transferred, archived, or used in the next workflow. The source study remains unchanged.',
      placement: 'left',
    },
  ],
};

export const AI_SEGMENTATION_REVIEW_TUTORIAL: TaskTutorialConfig = {
  templateId: 'segmentation-annotation',
  title: 'AI Segmentation Review Walkthrough',
  storageKey: 'segpro-med.workflow.aiSegmentationReviewTutorialDismissed',
  steps: [
    {
      nodeKey: 'loader',
      title: 'Load the image data',
      body: 'Start with Data Loader. When a file or folder is selected, this task automatically loads the study and opens it in Interactive Annotator so the image is visible before segmentation.',
      placement: 'right',
    },
    {
      nodeKey: 'annotator',
      title: 'Inspect the slice first',
      body: 'Use Interactive Annotator to pan, zoom, change view, and move through slices. The Batch SAM2 node follows the current view and slice from here.',
      placement: 'left',
    },
    {
      nodeKey: 'medsam2',
      title: 'Run Batch SAM2 deliberately',
      body: 'Batch SAM2 Segmenter uses the loaded session and current annotator view/slice. Review the visible image first, then run this node when the segmentation settings are correct.',
      placement: 'left',
    },
  ],
};

export const MEDICAL_REPORT_GENERATION_TUTORIAL: TaskTutorialConfig = {
  templateId: 'medgemma-slice-report',
  title: 'Medical Report Generation Walkthrough',
  storageKey: 'segpro-med.workflow.medicalReportGenerationTutorialDismissed',
  steps: [
    {
      nodeKey: 'loader',
      title: 'Load the study',
      body: 'Start with Data Loader. When a file or folder is selected, this task automatically loads the imaging data and sends it to Interactive Annotator.',
      placement: 'right',
    },
    {
      nodeKey: 'annotator',
      title: 'Choose the visible slice',
      body: 'Use Interactive Annotator to inspect the image, change view plane, and move to the slice you want reported. The report node follows this view and slice.',
      placement: 'right',
    },
    {
      nodeKey: 'medgemma',
      title: 'Generate the report',
      body: 'Medical Report Generation uses the current annotator context. Select the prompt, review model settings if needed, then create the report for the visible slice.',
      placement: 'left',
    },
  ],
};

export const MEDICAL_REPORT_OUTPUT_TUTORIAL: TaskTutorialConfig = {
  templateId: 'medgemma-slice-report',
  title: 'Medical Report Output Walkthrough',
  storageKey: 'segpro-med.workflow.medicalReportOutputTutorialDismissed',
  steps: [
    {
      nodeKey: 'medgemma',
      title: 'Review the generated report',
      body: 'The VLM output is now displayed inside this node. Click the report preview to open it in a larger modal for easier reading and interpretation.',
      placement: 'left',
    },
  ],
};

export const VLM_LABEL_SUGGESTIONS_TUTORIAL: TaskTutorialConfig = {
  templateId: 'vlm-label-suggestions',
  title: 'VLM Label Suggestions Walkthrough',
  storageKey: 'segpro-med.workflow.vlmLabelSuggestionsTutorialDismissed',
  steps: [
    {
      nodeKey: 'loader',
      title: 'Load the study',
      body: 'Start with Data Loader. Select the image file or folder that should be reviewed for label suggestions.',
      placement: 'right',
    },
    {
      nodeKey: 'annotator',
      title: 'Inspect the target slice',
      body: 'Use Interactive Annotator to view the image, change view plane, and move to the slice where labels should be suggested.',
      placement: 'right',
    },
    {
      nodeKey: 'labels',
      title: 'Generate label suggestions',
      body: 'Label Suggester uses the current image and slice context. Run this node only after the visible slice is the one you want reviewed.',
      placement: 'left',
    },
  ],
};

export const VLM_LABEL_SUGGESTIONS_OUTPUT_TUTORIAL: TaskTutorialConfig = {
  templateId: 'vlm-label-suggestions',
  title: 'Label Suggestions Approval',
  storageKey: 'segpro-med.workflow.vlmLabelSuggestionsOutputTutorialDismissed',
  steps: [
    {
      nodeKey: 'labels',
      title: 'Review generated labels',
      body: 'Labels have been generated and are waiting for your approval. Please review the suggested labels carefully. Approved labels will be saved for the currently selected image and slice.',
      placement: 'left',
    },
  ],
};

export const taskTutorials: Record<string, TaskTutorialConfig> = {
  [PHI_DEIDENTIFICATION_TUTORIAL.templateId]: PHI_DEIDENTIFICATION_TUTORIAL,
  [AI_SEGMENTATION_REVIEW_TUTORIAL.templateId]: AI_SEGMENTATION_REVIEW_TUTORIAL,
  [MEDICAL_REPORT_GENERATION_TUTORIAL.templateId]: MEDICAL_REPORT_GENERATION_TUTORIAL,
  [VLM_LABEL_SUGGESTIONS_TUTORIAL.templateId]: VLM_LABEL_SUGGESTIONS_TUTORIAL,
};
