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
import type { CSSProperties } from 'react';

interface TemplateNodeSpec {
  key: string;
  type: string;
  position: { x: number; y: number };
  data?: Record<string, unknown>;
  style?: CSSProperties;
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

const ANNOTATOR_INITIAL_STYLE = { width: 430, height: 560 };

export const workflowTemplates: WorkflowTemplate[] = [
  {
    id: 'segmentation-annotation',
    title: 'AI Segmentation Review',
    description: 'Load a study, choose a SAM2 profile, batch segment selected slices, then refine polygons in the annotator.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 130 } },
      { key: 'viewer', type: 'sliceViewer', position: { x: 430, y: 330 } },
      { key: 'profile', type: 'autoSegmentation', position: { x: 430, y: 80 } },
      { key: 'medsam2', type: 'medsam2Segmenter', position: { x: 790, y: 80 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 1150, y: 80 }, style: ANNOTATOR_INITIAL_STYLE },
    ],
    connections: [
      { source: 'loader', target: 'viewer' },
      { source: 'loader', target: 'medsam2' },
      { source: 'loader', target: 'annotator' },
      { source: 'profile', target: 'medsam2' },
      { source: 'profile', target: 'annotator' },
      { source: 'medsam2', target: 'annotator' },
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
    title: 'Manual Annotation Export',
    description: 'Load a study, annotate manually, store annotations, export JSON, and send the annotated session to a read-only viewer branch.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 130 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 430, y: 100 }, style: ANNOTATOR_INITIAL_STYLE },
      { key: 'viewer', type: 'sliceViewer', position: { x: 790, y: 180 } },
      { key: 'store', type: 'annotationStore', position: { x: 790, y: -40 } },
      { key: 'export', type: 'exportNode', position: { x: 1150, y: -40 } },
    ],
    connections: [
      { source: 'loader', target: 'annotator' },
      { source: 'annotator', target: 'viewer' },
      { source: 'annotator', target: 'store' },
      { source: 'store', target: 'export' },
    ],
  },
  {
    id: 'medgemma-slice-report',
    title: 'Medical Report Generation',
    description: 'Load a study, inspect the target slice in the annotator, then generate a report with MedGemma or another configured medical VLM.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 210 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 490, y: 150 }, style: ANNOTATOR_INITIAL_STYLE },
      { key: 'medgemma', type: 'medgemmaNode', position: { x: 930, y: 120 }, style: { width: 460, height: 500 } },
    ],
    connections: [
      { source: 'loader', target: 'annotator' },
      { source: 'annotator', target: 'medgemma' },
    ],
  },
  {
    id: 'vlm-label-suggestions',
    title: 'VLM Label Suggestions',
    description: 'Load a study, inspect the active slice in the annotator, then generate reviewable label suggestions for that slice.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 210 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 490, y: 150 }, style: ANNOTATOR_INITIAL_STYLE },
      { key: 'labels', type: 'labelSuggester', position: { x: 930, y: 120 }, style: { width: 460, height: 500 } },
    ],
    connections: [
      { source: 'loader', target: 'annotator' },
      { source: 'annotator', target: 'labels' },
    ],
  },
  {
    id: 'voice-guided-vlm-review',
    title: 'Voice-Guided VLM Review',
    description: 'Dictate a voice prompt, load and annotate a study, then drive medical report generation and label suggestions using the spoken instruction.',
    nodes: [
      { key: 'loader', type: 'dataLoader', position: { x: 80, y: 280 } },
      { key: 'voice', type: 'voiceInput', position: { x: 490, y: 60 } },
      { key: 'annotator', type: 'interactiveAnnotator', position: { x: 490, y: 270 }, style: ANNOTATOR_INITIAL_STYLE },
      { key: 'medgemma', type: 'medgemmaNode', position: { x: 930, y: 160 } },
      { key: 'labels', type: 'labelSuggester', position: { x: 1340, y: 160 } },
    ],
    connections: [
      { source: 'loader', target: 'annotator' },
      { source: 'annotator', target: 'medgemma' },
      { source: 'voice', target: 'medgemma' },
      { source: 'medgemma', target: 'labels' },
    ],
  },
  {
    id: 'collaboration-campaign',
    title: 'Collaboration Campaign',
    description: 'Create or reuse a campaign, assign unassigned patients to one expert, and check the refreshed campaign status.',
    nodes: [
      { key: 'setup', type: 'campaignSetup', position: { x: 80, y: 120 } },
      { key: 'assign', type: 'patientAssign', position: { x: 430, y: 120 } },
      { key: 'status', type: 'campaignStatus', position: { x: 790, y: 120 } },
    ],
    connections: [
      { source: 'setup', target: 'assign' },
      { source: 'assign', target: 'status' },
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
      style: spec.style,
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
