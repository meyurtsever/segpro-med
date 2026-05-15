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

    if (node.type === 'autoSegmentation') {
      const hasDataSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath']);
      if (!hasDataSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: 'Auto Segmentation needs a Data Loader or Format Converter input.',
        });
      }

      if (getOutgoingEdges(node.id, edges).length === 0) {
        issues.push({
          id: `${node.id}:output`,
          severity: 'warning',
          nodeId: node.id,
          message: 'Auto Segmentation is most useful when connected to Interactive Annotator.',
        });
      }
    }

    if (node.type === 'interactiveAnnotator') {
      const hasDataSource = hasValue(data.sessionId) ||
        hasIncomingKind(node, nodes, edges, ['session', 'filePath', 'segmentationConfig']);
      if (!hasDataSource) {
        issues.push({
          id: `${node.id}:session`,
          severity: 'error',
          nodeId: node.id,
          message: 'Interactive Annotator needs a data source before annotation.',
        });
      }
    }
  }

  return issues;
}
