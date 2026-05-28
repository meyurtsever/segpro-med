/**
 * Workflow Validation
 * ===================
 * Produces user-facing issues before workflow execution starts.
 *
 * Use errors for missing inputs that would make execution fail, and warnings
 * for guidance that improves the graph but should not block running. Keep these
 * checks aligned with nodeContracts.ts and executor.ts.
 */

import type { Edge, Node } from '@xyflow/react';

import type { BaseNodeData } from '../types/nodes';
import {
  getConnectionKinds,
  type WorkflowPortKind,
} from './nodeContracts';

export interface WorkflowIssue {
  id: string;
  severity: 'error' | 'warning';
  message: string;
  nodeId?: string;
}

function getIncomingEdges(nodeId: string, edges: Edge[]) {
  return edges.filter((edge) => edge.target === nodeId);
}

function getOutgoingEdges(nodeId: string, edges: Edge[]) {
  return edges.filter((edge) => edge.source === nodeId);
}

function hasIncomingKind(
  node: Node<BaseNodeData>,
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
  kinds: WorkflowPortKind[],
) {
  return getIncomingEdges(node.id, edges).some((edge) => {
    const source = nodes.find((candidate) => candidate.id === edge.source);
    return getConnectionKinds(source?.type, node.type).some((kind) => kinds.includes(kind));
  });
}

function hasValue(value: unknown) {
  return typeof value === 'string' ? value.trim().length > 0 : Boolean(value);
}

function getConverterInputPathIssue(conversionType: unknown, inputPath: unknown) {
  if (typeof conversionType !== 'string' || typeof inputPath !== 'string') {
    return undefined;
  }

  const normalizedPath = inputPath.trim().toLowerCase();
  if (!normalizedPath) return undefined;

  if (conversionType.startsWith('NIFTI') &&
    !normalizedPath.endsWith('.nii') &&
    !normalizedPath.endsWith('.nii.gz')
  ) {
    return 'NIfTI conversions require an input file ending in .nii or .nii.gz.';
  }

  if (conversionType.startsWith('DICOM') &&
    (normalizedPath.endsWith('.nii') || normalizedPath.endsWith('.nii.gz') || normalizedPath.endsWith('.mat'))
  ) {
    return 'DICOM conversions require a DICOM directory or a single .dcm file.';
  }

  return undefined;
}

export function validateWorkflow(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
): WorkflowIssue[] {
  const issues: WorkflowIssue[] = [];

  for (const node of nodes) {
    const data = node.data as Record<string, unknown>;

    if (node.type === 'dataLoader') {
      if (!hasValue(data.path)) {
        issues.push({
          id: `${node.id}:path`,
          severity: 'error',
          nodeId: node.id,
          message: 'Data Loader needs a file or directory path.',
        });
      }
    }

    if (node.type === 'formatConverter') {
      if (!hasValue(data.conversionType)) {
        issues.push({
          id: `${node.id}:conversionType`,
          severity: 'error',
          nodeId: node.id,
          message: 'Format Converter needs a conversion type.',
        });
      }

      const hasInputPath = hasValue(data.inputPath) ||
        hasIncomingKind(node, nodes, edges, ['filePath']);
      if (!hasInputPath) {
        issues.push({
          id: `${node.id}:inputPath`,
          severity: 'error',
          nodeId: node.id,
          message: 'Format Converter needs an input path or a file-path connection.',
        });
      }

      const pathIssue = getConverterInputPathIssue(data.conversionType, data.inputPath);
      if (pathIssue) {
        issues.push({
          id: `${node.id}:inputPathType`,
          severity: 'error',
          nodeId: node.id,
          message: pathIssue,
        });
      }
    }

    if (node.type === 'sliceViewer') {
      const hasDataSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath', 'annotatedSession']);
      if (!hasDataSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: 'Slice Viewer needs a Data Loader or Format Converter input.',
        });
      }
    }

    if (node.type === 'metadataViewer') {
      const hasMetadataSource = hasValue(data.sessionId) ||
        hasValue(data.sourcePath) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath']);
      if (!hasMetadataSource) {
        issues.push({
          id: `${node.id}:source`,
          severity: 'error',
          nodeId: node.id,
          message: 'Metadata Viewer needs a Data Loader, Format Converter, or source path.',
        });
      }
    }

    if (node.type === 'autoSegmentation') {
      if (getOutgoingEdges(node.id, edges).length === 0) {
        issues.push({
          id: `${node.id}:output`,
          severity: 'warning',
          nodeId: node.id,
          message: 'Segmentation Profile is most useful when connected to Interactive Annotator or Batch SAM2 Segmenter.',
        });
      }
    }

    if (node.type === 'medsam2Segmenter') {
      const hasDataSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath']);
      if (!hasDataSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: 'Batch SAM2 Segmenter needs a Data Loader or Format Converter input.',
        });
      }
      if (data.promptMode === 'prompt' &&
        !hasValue(data.segmentationPrompt) &&
        !hasIncomingKind(node, nodes, edges, ['segmentationPrompt'])
      ) {
        issues.push({
          id: `${node.id}:prompt`,
          severity: 'error',
          nodeId: node.id,
          message: 'Prompt SAM2 mode needs a point or rectangle prompt from Interactive Annotator.',
        });
      }
    }

    if (node.type === 'interactiveAnnotator') {
      const hasDataSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath', 'segmentationResult', 'annotatedSession']);
      if (!hasDataSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: 'Interactive Annotator needs a data source before annotation.',
        });
      }
    }

    if (node.type === 'deidentifyNode') {
      const hasSource = hasValue(data.sessionId) ||
        hasValue(data.sourcePath) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath']);
      if (!hasSource) {
        issues.push({
          id: `${node.id}:source`,
          severity: 'error',
          nodeId: node.id,
          message: 'Deidentify needs a Data Loader connection or source path.',
        });
      }
    }

    if (node.type === 'annotationStore') {
      const hasAnnotations = hasIncomingKind(node, nodes, edges, ['annotatedSession']);
      const hasStudyPath = hasValue(data.studyPath);
      if (!hasAnnotations) {
        issues.push({
          id: `${node.id}:annotations`,
          severity: 'error',
          nodeId: node.id,
          message: 'Annotation Store needs an annotated-session input.',
        });
      }
      if (!hasStudyPath && !hasAnnotations) {
        issues.push({
          id: `${node.id}:studyPath`,
          severity: 'warning',
          nodeId: node.id,
          message: 'Annotation Store needs a study path from configuration or upstream annotator data.',
        });
      }
    }

    if (node.type === 'annotationLoad') {
      const hasStudyPath = hasValue(data.studyPath) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath']);
      if (!hasStudyPath) {
        issues.push({
          id: `${node.id}:studyPath`,
          severity: 'error',
          nodeId: node.id,
          message: 'Annotation Load needs a study path or data-source connection.',
        });
      }
    }

    if (node.type === 'exportNode') {
      const hasExportSource = hasIncomingKind(node, nodes, edges, ['annotatedSession', 'annotationRecord', 'filePath', 'session', 'deidentifiedData']);
      if (!hasExportSource) {
        issues.push({
          id: `${node.id}:source`,
          severity: 'error',
          nodeId: node.id,
          message: 'Export needs annotations, an annotation record, or a data output.',
        });
      }
    }

    if (node.type === 'medgemmaNode' || node.type === 'smolvlmNode' || node.type === 'medR1Node') {
      const hasImageSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath', 'annotatedSession']);
      if (!hasImageSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: `${node.data.label || 'VLM node'} needs a loaded image source.`,
        });
      }
      if (
        !hasValue(data.promptKey) &&
        !hasValue(data.customPrompt) &&
        !hasIncomingKind(node, nodes, edges, ['voicePrompt'])
      ) {
        issues.push({
          id: `${node.id}:prompt`,
          severity: 'error',
          nodeId: node.id,
          message: `${node.data.label || 'VLM node'} needs a prompt preset or custom prompt.`,
        });
      }
    }

    if (node.type === 'voiceInput') {
      if (!hasValue(data.transcript) && !hasValue(data.audioPath)) {
        issues.push({
          id: `${node.id}:voice`,
          severity: 'error',
          nodeId: node.id,
          message: 'Voice Input needs a transcript or audio file path.',
        });
      }
    }

    if (node.type === 'labelSuggester') {
      const hasSuggestionSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath', 'annotatedSession', 'vlmAnalysis']);
      if (!hasSuggestionSource) {
        issues.push({
          id: `${node.id}:source`,
          severity: 'error',
          nodeId: node.id,
          message: 'Label Suggester needs an image source or VLM analysis input.',
        });
      }
    }

    if (node.type === 'campaignSetup') {
      if (!hasValue(data.campaignName)) {
        issues.push({
          id: `${node.id}:campaignName`,
          severity: 'error',
          nodeId: node.id,
          message: 'Campaign Setup needs a campaign name.',
        });
      }
      if (!hasValue(data.datasetPath)) {
        issues.push({
          id: `${node.id}:datasetPath`,
          severity: 'error',
          nodeId: node.id,
          message: 'Campaign Setup needs a dataset root path.',
        });
      }
    }

    if (node.type === 'patientAssign') {
      const hasCampaign = hasValue(data.campaignName) ||
        hasIncomingKind(node, nodes, edges, ['campaign']);
      if (!hasCampaign) {
        issues.push({
          id: `${node.id}:campaign`,
          severity: 'error',
          nodeId: node.id,
          message: 'Patient Assign needs a campaign input or campaign name.',
        });
      }
      if (!hasValue(data.expertId)) {
        issues.push({
          id: `${node.id}:expert`,
          severity: 'error',
          nodeId: node.id,
          message: 'Patient Assign needs an expert ID.',
        });
      }
      if (data.assignmentMode === 'selected' && !hasValue(data.patientIdsText)) {
        issues.push({
          id: `${node.id}:patients`,
          severity: 'error',
          nodeId: node.id,
          message: 'Selected patient assignment needs patient IDs.',
        });
      }
    }

    if (node.type === 'campaignStatus') {
      const hasCampaign = hasValue(data.campaignName) ||
        hasIncomingKind(node, nodes, edges, ['campaign', 'assignment']);
      if (!hasCampaign) {
        issues.push({
          id: `${node.id}:campaign`,
          severity: 'error',
          nodeId: node.id,
          message: 'Campaign Status needs a campaign or assignment input.',
        });
      }
    }
  }

  return issues;
}
