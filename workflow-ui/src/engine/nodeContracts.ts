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
  | 'segmentationConfig'
  | 'annotatedSession';

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
  segmentationConfig: {
    kind: 'segmentationConfig',
    label: 'Segmentation config',
    edgeLabel: 'seg config',
    color: 'var(--accent-purple)',
  },
  annotatedSession: {
    kind: 'annotatedSession',
    label: 'Annotated session',
    edgeLabel: 'annotations',
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
    inputKinds: ['session', 'filePath', 'segmentationConfig'],
    outputKinds: ['annotatedSession'],
    defaultData: {
      label: 'Interactive Annotator',
      status: 'idle',
      sessionId: '',
      sliceIndex: 0,
      view: 'axial',
      totalSlices: 0,
      annotations: [],
      sliceAnnotationsMap: {},
      activeTool: 'rect',
      zoom: 1,
      showLabels: true,
    },
  },
  {
    type: 'autoSegmentation',
    label: 'Auto Segmentation',
    icon: '🔬',
    category: 'Segmentation',
    categoryColor: 'var(--accent-purple)',
    description: 'SAM2 automatic ROI detection - connect to Interactive Annotator',
    inputKinds: ['session', 'filePath'],
    outputKinds: ['segmentationConfig'],
    defaultData: {
      label: 'Auto Segmentation',
      status: 'idle',
      sessionId: '',
      configName: 'fast',
      availableConfigs: [],
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
