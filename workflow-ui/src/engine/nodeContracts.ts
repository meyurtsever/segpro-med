/**
 * Workflow Node Contracts
 * =======================
 * Central source of truth for workflow node metadata and data compatibility.
 *
 * Add new node types and port kinds here first. The palette, quick-add menu,
 * edge labels, inspector guidance, and connection validation all derive from
 * these contracts so new nodes behave consistently across the workflow UI.
 */

import type { BaseNodeData } from '../types/nodes';

export type WorkflowPortKind =
  | 'filePath'
  | 'session'
  | 'deidentifiedData'
  | 'segmentationConfig'
  | 'segmentationPrompt'
  | 'segmentationResult'
  | 'annotatedSession'
  | 'annotationRecord'
  | 'vlmAnalysis'
  | 'labelSuggestions'
  | 'voicePrompt'
  | 'campaign'
  | 'assignment';

export interface WorkflowPortDefinition {
  kind: WorkflowPortKind;
  label: string;
  edgeLabel: string;
  color: string;
}

export interface WorkflowNodeContract {
  type: string;
  label: string;
  icon: string;
  category: string;
  categoryColor: string;
  description: string;
  inputKinds: WorkflowPortKind[];
  outputKinds: WorkflowPortKind[];
  defaultData: BaseNodeData;
}

export const portDefinitions: Record<WorkflowPortKind, WorkflowPortDefinition> = {
  filePath: {
    kind: 'filePath',
    label: 'File path',
    edgeLabel: 'file path',
    color: 'var(--accent-orange)',
  },
  session: {
    kind: 'session',
    label: 'Loaded session',
    edgeLabel: 'session',
    color: 'var(--accent-blue)',
  },
  deidentifiedData: {
    kind: 'deidentifiedData',
    label: 'Deidentified data',
    edgeLabel: 'deidentified',
    color: 'var(--accent-green)',
  },
  segmentationConfig: {
    kind: 'segmentationConfig',
    label: 'Segmentation config',
    edgeLabel: 'seg config',
    color: 'var(--accent-purple)',
  },
  segmentationPrompt: {
    kind: 'segmentationPrompt',
    label: 'SAM2 prompt',
    edgeLabel: 'prompt',
    color: 'var(--accent-orange)',
  },
  segmentationResult: {
    kind: 'segmentationResult',
    label: 'Segmentation result',
    edgeLabel: 'seg result',
    color: 'var(--accent-green)',
  },
  annotatedSession: {
    kind: 'annotatedSession',
    label: 'Annotated session',
    edgeLabel: 'annotations',
    color: 'var(--accent-green)',
  },
  annotationRecord: {
    kind: 'annotationRecord',
    label: 'Annotation record',
    edgeLabel: 'record',
    color: 'var(--accent-blue)',
  },
  vlmAnalysis: {
    kind: 'vlmAnalysis',
    label: 'VLM analysis',
    edgeLabel: 'vlm',
    color: 'var(--accent-purple)',
  },
  labelSuggestions: {
    kind: 'labelSuggestions',
    label: 'Label suggestions',
    edgeLabel: 'labels',
    color: 'var(--accent-green)',
  },
  voicePrompt: {
    kind: 'voicePrompt',
    label: 'Voice prompt',
    edgeLabel: 'voice',
    color: 'var(--accent-orange)',
  },
  campaign: {
    kind: 'campaign',
    label: 'Campaign',
    edgeLabel: 'campaign',
    color: 'var(--accent-blue)',
  },
  assignment: {
    kind: 'assignment',
    label: 'Assignment',
    edgeLabel: 'assignment',
    color: 'var(--accent-green)',
  },
};

export const nodeContracts: WorkflowNodeContract[] = [
  {
    type: 'dataLoader',
    label: 'Data Loader',
    icon: '📂',
    category: 'Data I/O',
    categoryColor: 'var(--accent-blue)',
    description: 'Load DICOM / NIfTI / MAT from local path',
    inputKinds: [],
    outputKinds: ['session', 'filePath'],
    defaultData: {
      label: 'Data Loader',
      status: 'idle',
      path: '',
    },
  },
  {
    type: 'formatConverter',
    label: 'Format Converter',
    icon: '🔄',
    category: 'Data I/O',
    categoryColor: 'var(--accent-orange)',
    description: 'Convert between DICOM, NIfTI, MAT, PNG',
    inputKinds: ['filePath'],
    outputKinds: ['filePath'],
    defaultData: {
      label: 'Format Converter',
      status: 'idle',
      inputPath: '',
      outputPath: '',
      conversionType: '',
      axis: 2,
      compress: true,
    },
  },
  {
    type: 'metadataViewer',
    label: 'Metadata Viewer',
    icon: 'i',
    category: 'Data I/O',
    categoryColor: 'var(--accent-blue)',
    description: 'Inspect DICOM/NIfTI/MAT metadata from a session or file path',
    inputKinds: ['session', 'filePath'],
    outputKinds: [],
    defaultData: {
      label: 'Metadata Viewer',
      status: 'idle',
      sourcePath: '',
      filterText: '',
    },
  },
  {
    type: 'deidentifyNode',
    label: 'Deidentify',
    icon: 'ID',
    category: 'Data I/O',
    categoryColor: 'var(--accent-green)',
    description: 'Blank DICOM PHI fields and deface 3D volumes with an audit trail',
    inputKinds: ['session', 'filePath'],
    outputKinds: ['session', 'filePath', 'deidentifiedData'],
    defaultData: {
      label: 'Deidentify',
      status: 'idle',
      sessionId: '',
      sourcePath: '',
      outputPath: '',
      applyDeface: true,
      sanitizedFieldCount: 0,
      filesProcessed: 0,
    },
  },
  {
    type: 'sliceViewer',
    label: 'Slice Viewer',
    icon: '🖼️',
    category: 'Visualization',
    categoryColor: 'var(--accent-purple)',
    description: 'View image slices with navigation controls',
    inputKinds: ['session', 'filePath', 'annotatedSession'],
    outputKinds: [],
    defaultData: {
      label: 'Slice Viewer',
      status: 'idle',
      sessionId: '',
      sliceIndex: 0,
      view: 'axial',
      totalSlices: 0,
      zoom: 1,
    },
  },
  {
    type: 'interactiveAnnotator',
    label: 'Interactive Annotator',
    icon: '✏️',
    category: 'Annotation',
    categoryColor: 'var(--accent-green)',
    description: 'Draw polygons, circles, freehand paths & coordinate markers on slices',
    inputKinds: ['session', 'filePath', 'segmentationConfig', 'segmentationResult', 'annotatedSession', 'labelSuggestions'],
    outputKinds: ['annotatedSession', 'segmentationPrompt'],
    defaultData: {
      label: 'Interactive Annotator',
      status: 'idle',
      sessionId: '',
      sliceIndex: 0,
      view: 'axial',
      totalSlices: 0,
      annotations: [],
      sliceAnnotationsMap: {},
      activeTool: 'pan',
      zoom: 1,
      showLabels: true,
    },
  },
  {
    type: 'autoSegmentation',
    label: 'Segmentation Profile',
    icon: 'CFG',
    category: 'Segmentation',
    categoryColor: 'var(--accent-purple)',
    description: 'Choose the SAM2 profile used by Annotator or Batch SAM2 Segmenter',
    inputKinds: [],
    outputKinds: ['segmentationConfig'],
    defaultData: {
      label: 'Segmentation Profile',
      status: 'idle',
      configName: 'fast',
      availableConfigs: [],
    },
  },
  {
    type: 'medsam2Segmenter',
    label: 'Batch SAM2 Segmenter',
    icon: 'S2',
    category: 'Segmentation',
    categoryColor: 'var(--accent-purple)',
    description: 'Run SAM2 over one slice, a slice range, or the whole scan',
    inputKinds: ['session', 'filePath', 'segmentationConfig', 'segmentationPrompt'],
    outputKinds: ['segmentationResult'],
    defaultData: {
      label: 'Batch SAM2 Segmenter',
      status: 'idle',
      sessionId: '',
      runMode: 'single',
      sliceIndex: 0,
      sliceStart: 0,
      sliceEnd: 0,
      sliceStep: 1,
      view: 'axial',
      configName: 'fast',
      promptMode: 'auto',
      availableConfigs: [],
    },
  },
  {
    type: 'annotationStore',
    label: 'Annotation Store',
    icon: 'AS',
    category: 'Annotation',
    categoryColor: 'var(--accent-green)',
    description: 'Persist annotations for a user and study through the annotation store',
    inputKinds: ['annotatedSession'],
    outputKinds: ['annotationRecord'],
    defaultData: {
      label: 'Annotation Store',
      status: 'idle',
      userId: 'workflow_user',
      studyPath: '',
      annotationType: 'manual',
    },
  },
  {
    type: 'annotationLoad',
    label: 'Annotation Load',
    icon: 'AL',
    category: 'Annotation',
    categoryColor: 'var(--accent-green)',
    description: 'Load persisted annotations and send them back into the workflow',
    inputKinds: ['session', 'filePath'],
    outputKinds: ['annotationRecord', 'annotatedSession'],
    defaultData: {
      label: 'Annotation Load',
      status: 'idle',
      userId: 'workflow_user',
      studyPath: '',
      annotations: [],
      sliceAnnotationsMap: {},
    },
  },
  {
    type: 'exportNode',
    label: 'Export',
    icon: 'EX',
    category: 'Data I/O',
    categoryColor: 'var(--accent-orange)',
    description: 'Export stored annotation records to a workflow output file',
    inputKinds: ['annotatedSession', 'annotationRecord', 'filePath', 'session', 'deidentifiedData'],
    outputKinds: ['filePath'],
    defaultData: {
      label: 'Export',
      status: 'idle',
      userId: 'workflow_user',
      studyPath: '',
      exportFormat: 'json',
      outputPath: '',
    },
  },
  {
    type: 'voiceInput',
    label: 'Voice Input',
    icon: 'VO',
    category: 'VLM',
    categoryColor: 'var(--accent-orange)',
    description: 'Capture or transcribe a spoken instruction for medical VLM analysis',
    inputKinds: [],
    outputKinds: ['voicePrompt'],
    defaultData: {
      label: 'Voice Input',
      status: 'idle',
      transcript: '',
      audioPath: '',
      audioFileName: '',
      autoDetectIntent: true,
      intent: 'both',
      promptKey: 'structured_radiology_review',
    },
  },
  {
    type: 'medgemmaNode',
    label: 'Medical Report Generation',
    icon: 'MR',
    category: 'VLM',
    categoryColor: 'var(--accent-purple)',
    description: 'Generate a medical image report with MedGemma GGUF by default, or switch VLM model in settings',
    inputKinds: ['session', 'filePath', 'annotatedSession', 'voicePrompt'],
    outputKinds: ['vlmAnalysis'],
    defaultData: {
      label: 'Medical Report Generation',
      status: 'idle',
      sessionId: '',
      sliceIndex: 0,
      view: 'axial',
      model: 'medgemma-1.5-gguf',
      modality: 'MRI',
      promptKey: 'describe_slice',
      customPrompt: '',
      maxTokens: 512,
      includeReasoning: false,
      useOverlay: false,
      availablePrompts: [],
    },
  },
  {
    type: 'labelSuggester',
    label: 'Label Suggester',
    icon: 'LS',
    category: 'VLM',
    categoryColor: 'var(--accent-green)',
    description: 'Generate concise labels from a slice, annotations, or VLM analysis',
    inputKinds: ['session', 'filePath', 'annotatedSession', 'vlmAnalysis', 'voicePrompt'],
    outputKinds: ['labelSuggestions'],
    defaultData: {
      label: 'Label Suggester',
      status: 'idle',
      sessionId: '',
      sliceIndex: 0,
      view: 'axial',
      model: 'medgemma-1.5-gguf',
      modality: 'MRI',
      promptKey: 'suggest_labels',
      customPrompt: '',
      maxTokens: 512,
      maxLabels: 12,
      useOverlay: true,
      availablePrompts: [],
      currentLabels: [],
      labelSuggestions: [],
    },
  },
  {
    type: 'campaignSetup',
    label: 'Campaign Setup',
    icon: 'CS',
    category: 'Collaboration',
    categoryColor: 'var(--accent-blue)',
    description: 'Create or reuse a crowdsourcing campaign from a dataset root',
    inputKinds: [],
    outputKinds: ['campaign'],
    defaultData: {
      label: 'Campaign Setup',
      status: 'idle',
      campaignName: '',
      datasetPath: '',
      description: '',
    },
  },
  {
    type: 'patientAssign',
    label: 'Patient Assign',
    icon: 'PA',
    category: 'Collaboration',
    categoryColor: 'var(--accent-green)',
    description: 'Assign campaign patients to one expert annotator',
    inputKinds: ['campaign'],
    outputKinds: ['assignment'],
    defaultData: {
      label: 'Patient Assign',
      status: 'idle',
      campaignName: '',
      expertId: '',
      assignmentMode: 'allUnassigned',
      patientIdsText: '',
      availableExperts: [],
      unassignedPatients: [],
    },
  },
  {
    type: 'campaignStatus',
    label: 'Campaign Status',
    icon: 'ST',
    category: 'Collaboration',
    categoryColor: 'var(--accent-blue)',
    description: 'Inspect campaign progress across assigned, completed, and reviewed patients',
    inputKinds: ['campaign', 'assignment'],
    outputKinds: [],
    defaultData: {
      label: 'Campaign Status',
      status: 'idle',
      campaignName: '',
    },
  },
  {
    type: 'crowdsourcingTasks',
    label: 'Assignment Tasks',
    icon: 'AT',
    category: 'Collaboration',
    categoryColor: 'var(--accent-green)',
    description: 'Show assigned crowdsourcing tasks and current task guidance for experts',
    inputKinds: [],
    outputKinds: [],
    defaultData: {
      label: 'Assignment Tasks',
      status: 'idle',
      userId: '',
      currentPatientId: '',
      tasks: [],
    },
  },
];

export const nodePaletteItems = nodeContracts.map((contract) => ({
  type: contract.type,
  label: contract.label,
  icon: contract.icon,
  category: contract.category,
  categoryColor: contract.categoryColor,
  description: contract.description,
  defaultData: contract.defaultData,
}));

export function getNodeContract(type: string | undefined): WorkflowNodeContract | undefined {
  return nodeContracts.find((contract) => contract.type === type);
}

export function formatNodeType(type: string | undefined): string {
  return getNodeContract(type)?.label || type || 'Unknown';
}

export function getPortDefinition(kind: WorkflowPortKind | undefined) {
  return kind ? portDefinitions[kind] : undefined;
}

export function getConnectionKinds(
  sourceType: string | undefined,
  targetType: string | undefined,
): WorkflowPortKind[] {
  const source = getNodeContract(sourceType);
  const target = getNodeContract(targetType);
  if (!source || !target) return [];

  return source.outputKinds.filter((kind) => target.inputKinds.includes(kind));
}

export function getConnectionEdgeData(
  sourceType: string | undefined,
  targetType: string | undefined,
) {
  const kind = getConnectionKinds(sourceType, targetType)[0];
  const definition = getPortDefinition(kind);
  if (!kind || !definition) return undefined;

  return {
    kind,
    label: definition.edgeLabel,
    color: definition.color,
  };
}

export function getAllowedTargets(sourceType: string): string[] {
  return nodeContracts
    .filter((contract) => getConnectionKinds(sourceType, contract.type).length > 0)
    .map((contract) => contract.type);
}

export function getAllowedSources(targetType: string): string[] {
  return nodeContracts
    .filter((contract) => getConnectionKinds(contract.type, targetType).length > 0)
    .map((contract) => contract.type);
}
