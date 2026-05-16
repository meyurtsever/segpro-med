/**
 * Workflow Persistence
 * ====================
 * Serializes and restores editable React Flow workflow graphs.
 *
 * This is intentionally graph-focused: saved files preserve node configuration,
 * positions, and typed connections, while transient runtime fields such as
 * execution status and rendered slice images are removed.
 */

import type { Edge, Node } from '@xyflow/react';

import type { BaseNodeData } from '../types/nodes';
import { getConnectionEdgeData, getNodeContract } from './nodeContracts';

export const WORKFLOW_STORAGE_KEY = 'segpro-med.workflow.lastSaved';

export interface WorkflowSnapshot {
  schemaVersion: 1;
  app: 'segpro-med-workflow';
  exportedAt: string;
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
}

const runtimeDataKeys = new Set([
  'status',
  'error',
  'imageBase64',
  'filePath',
  'sessionId',
  'fileType',
  'volumeShape',
  'metadata',
  'totalSlices',
  'availableConfigs',
  'outputSizeBytes',
  'previewSessionId',
  'previewFileType',
  'previewImageBase64',
  'previewTotalSlices',
  'previewVolumeShape',
  'previewMetadata',
]);

function cloneData(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== 'object') return {};
  return JSON.parse(JSON.stringify(data)) as Record<string, unknown>;
}

function cleanNodeData(type: string | undefined, data: unknown): BaseNodeData {
  const contract = getNodeContract(type);
  const defaults = cloneData(contract?.defaultData);
  const current = cloneData(data);

  for (const key of runtimeDataKeys) {
    delete current[key];
  }

  return {
    ...defaults,
    ...current,
    label: String(current.label || defaults.label || type || 'Node'),
    status: 'idle',
  } as BaseNodeData;
}

function cleanNode(node: Node<BaseNodeData>): Node<BaseNodeData> | null {
  if (!node.id || !node.type) return null;

  return {
    id: String(node.id),
    type: String(node.type),
    position: {
      x: Number(node.position?.x ?? 0),
      y: Number(node.position?.y ?? 0),
    },
    data: cleanNodeData(node.type, node.data),
    width: node.width,
    height: node.height,
  };
}

function cleanEdge(edge: Edge, nodes: Node<BaseNodeData>[]): Edge | null {
  if (!edge.source || !edge.target) return null;
  if (!nodes.some((node) => node.id === edge.source)) return null;
  if (!nodes.some((node) => node.id === edge.target)) return null;

  const sourceNode = nodes.find((node) => node.id === edge.source);
  const targetNode = nodes.find((node) => node.id === edge.target);

  return {
    id: edge.id || `edge_${edge.source}_${edge.target}`,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle ?? null,
    targetHandle: edge.targetHandle ?? null,
    type: 'typed',
    animated: true,
    data: getConnectionEdgeData(sourceNode?.type, targetNode?.type),
  };
}

export function createWorkflowSnapshot(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
): WorkflowSnapshot {
  const cleanNodes = nodes
    .map((node) => cleanNode(node))
    .filter((node): node is Node<BaseNodeData> => Boolean(node));
  const cleanEdges = edges
    .map((edge) => cleanEdge(edge, cleanNodes))
    .filter((edge): edge is Edge => Boolean(edge));

  return {
    schemaVersion: 1,
    app: 'segpro-med-workflow',
    exportedAt: new Date().toISOString(),
    nodes: cleanNodes,
    edges: cleanEdges,
  };
}

export function serializeWorkflowSnapshot(snapshot: WorkflowSnapshot): string {
  return JSON.stringify(snapshot, null, 2);
}

export function parseWorkflowSnapshot(raw: string): WorkflowSnapshot {
  const parsed = JSON.parse(raw) as Partial<WorkflowSnapshot>;
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Workflow file is not a valid JSON object.');
  }
  if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new Error('Workflow file must contain nodes and edges arrays.');
  }

  const snapshot = createWorkflowSnapshot(
    parsed.nodes as Node<BaseNodeData>[],
    parsed.edges as Edge[],
  );

  return {
    ...snapshot,
    exportedAt: typeof parsed.exportedAt === 'string' ? parsed.exportedAt : snapshot.exportedAt,
  };
}
