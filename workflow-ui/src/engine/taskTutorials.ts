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

export const taskTutorials: Record<string, TaskTutorialConfig> = {
  [PHI_DEIDENTIFICATION_TUTORIAL.templateId]: PHI_DEIDENTIFICATION_TUTORIAL,
};
