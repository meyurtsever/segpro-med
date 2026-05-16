/**
 * Workflow Templates
 * ==================
 * Defines reusable, editable starter graphs made from normal atomic nodes.
 *
 * Templates do not create special locked nodes. They instantiate the same node
 * contracts used by the palette, quick-add menu, validation, and typed edges.
 */

import type { Connection, Node } from '@xyflow/react';

import type { BaseNodeData } from '../types/nodes';
import { getNodeContract } from './nodeContracts';

interface TemplateNodeSpec {
  key: string;
  type: string;
  position: { x: number; y: number };
  data?: Record<string, unknown>;
}

interface TemplateConnectionSpec {
  source: string;
  target: string;
}

export interface WorkflowTemplate {
  id: string;
  title: string;
  description: string;
  nodes: TemplateNodeSpec[];
  connections: TemplateConnectionSpec[];
}

export interface TemplateInstance {
  nodes: Node<BaseNodeData>[];
  connections: Connection[];
}

export const workflowTemplates: WorkflowTemplate[] = [
  {
    id: 'segmentation-annotation',
    title: 'Segmentation Annotation',
    description: 'Load a study, preview slices, configure SAM2, then annotate with auto-segmentation support.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 130 } },
      { key: 'viewer', type: 'sliceViewer', position: { x: 430, y: 330 } },
      { key: 'segmenter', type: 'autoSegmentation', position: { x: 430, y: 80 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 790, y: 80 } },
    ],
    connections: [
      { source: 'loader', target: 'viewer' },
      { source: 'loader', target: 'segmenter' },
      { source: 'segmenter', target: 'annotator' },
    ],
  },
  {
    id: 'conversion-preview',
    title: 'Convert And Preview',
    description: 'Load imaging data, convert the file format, then preview and inspect the converted output.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 130 } },
      { key: 'converter', type: 'formatConverter', position: { x: 430, y: 130 } },
      { key: 'viewer', type: 'sliceViewer', position: { x: 790, y: 130 } },
      { key: 'metadata', type: 'metadataViewer', position: { x: 790, y: 430 } },
    ],
    connections: [
      { source: 'loader', target: 'converter' },
      { source: 'converter', target: 'viewer' },
      { source: 'converter', target: 'metadata' },
    ],
  },
  {
    id: 'manual-annotation-review',
    title: 'Manual Annotation Review',
    description: 'Load a study, annotate manually, then send the annotated session to a read-only viewer branch.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 130 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 430, y: 100 } },
      { key: 'viewer', type: 'sliceViewer', position: { x: 790, y: 180 } },
    ],
    connections: [
      { source: 'loader', target: 'annotator' },
      { source: 'annotator', target: 'viewer' },
    ],
  },
];

function cloneDefaultData(type: string) {
  const contract = getNodeContract(type);
  return JSON.parse(JSON.stringify(contract?.defaultData || {
    label: type,
    status: 'idle',
  })) as BaseNodeData;
}

export function instantiateWorkflowTemplate(
  template: WorkflowTemplate,
  createNodeId: () => string,
  offset: { x: number; y: number } = { x: 0, y: 0 },
): TemplateInstance {
  const idByKey = new Map<string, string>();
  const nodes = template.nodes.map((spec) => {
    const id = createNodeId();
    idByKey.set(spec.key, id);

    return {
      id,
      type: spec.type,
      position: {
        x: spec.position.x + offset.x,
        y: spec.position.y + offset.y,
      },
      data: {
        ...cloneDefaultData(spec.type),
        ...(spec.data || {}),
      },
    };
  });

  const connections = template.connections
    .map((connection) => ({
      source: idByKey.get(connection.source) || '',
      target: idByKey.get(connection.target) || '',
      sourceHandle: null,
      targetHandle: null,
    }))
    .filter((connection) => connection.source && connection.target);

  return { nodes, connections };
}
