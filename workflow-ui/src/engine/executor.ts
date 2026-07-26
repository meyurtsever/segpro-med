/**
 * Workflow Execution Engine
 * =========================
 * Executes React Flow graphs with branch-aware scheduling.
 *
 * Independent branches run in parallel when their dependencies are ready.
 * If one branch fails, only its downstream descendants are skipped; unrelated
 * branches can still complete. Scoped execution supports full workflow,
 * selected-node, and downstream-branch runs.
 */

import type { Edge, Node } from '@xyflow/react';
import type {
  AnnotationRecord,
  AnnotationShape,
  BaseNodeData,
  CampaignInfo,
  ExpertAssignment,
  LabelSuggestionResult,
  PatientAssignmentResult,
  SegmentationPrompt,
  SegmentationResult,
  SegmentationRunMode,
  SegmentationSliceResult,
  SliceAnnotationsMap,
  VlmAnalysisResult,
  VlmModelId,
  VlmModality,
  VoicePrompt,
  VoicePromptIntent,
} from '../types/nodes';
import * as api from '../api/client';

export type WorkflowRunMode = 'all' | 'selected' | 'downstream';

export interface ExecuteWorkflowOptions {
  mode?: WorkflowRunMode;
  startNodeId?: string;
  signal?: AbortSignal;
}

/** Results stored per node after execution */
export interface NodeResult {
  sessionId?: string;
  outputPath?: string;
  filePath?: string;
  [key: string]: unknown;
}

export interface WorkflowExecutionResult {
  mode: WorkflowRunMode;
  nodeIds: string[];
  executedNodeIds: string[];
  failedNodeIds: string[];
  skippedNodeIds: string[];
  results: Map<string, NodeResult>;
}

export class WorkflowExecutionCancelledError extends Error {
  constructor(message = 'Workflow run cancelled.') {
    super(message);
    this.name = 'WorkflowExecutionCancelledError';
  }
}

export class WorkflowExecutionFailedError extends Error {
  executedNodeIds: string[];
  failedNodeIds: string[];
  skippedNodeIds: string[];

  constructor(
    message: string,
    executedNodeIds: string[],
    failedNodeIds: string[],
    skippedNodeIds: string[],
  ) {
    super(message);
    this.name = 'WorkflowExecutionFailedError';
    this.executedNodeIds = executedNodeIds;
    this.failedNodeIds = failedNodeIds;
    this.skippedNodeIds = skippedNodeIds;
  }
}

export function isWorkflowExecutionCancelledError(
  error: unknown,
): error is WorkflowExecutionCancelledError {
  return error instanceof WorkflowExecutionCancelledError ||
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError');
}

function assertNotAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new WorkflowExecutionCancelledError();
  }
}

function getDownstreamNodeIds(startNodeId: string, edges: Edge[]) {
  const downstream = new Set<string>([startNodeId]);
  const queue = [startNodeId];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) continue;

    for (const edge of edges) {
      if (edge.source !== current || downstream.has(edge.target)) continue;
      downstream.add(edge.target);
      queue.push(edge.target);
    }
  }

  return downstream;
}

function getScopedNodeIds(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
  options: ExecuteWorkflowOptions,
) {
  const mode = options.mode || 'all';
  if (mode === 'all') {
    return new Set(nodes.map((node) => node.id));
  }

  if (!options.startNodeId || !nodes.some((node) => node.id === options.startNodeId)) {
    throw new Error('Select a node before running this action.');
  }

  return mode === 'selected'
    ? new Set([options.startNodeId])
    : getDownstreamNodeIds(options.startNodeId, edges);
}

function getNodeLabel(node: Node<BaseNodeData>) {
  return node.data.label || node.type || node.id;
}

function isVlmExecutionNode(type: string | undefined) {
  return type === 'medgemmaNode' || type === 'smolvlmNode' || type === 'medR1Node';
}

function isWaitingExecutionNode(
  node: Node<BaseNodeData>,
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
) {
  if (!isVlmExecutionNode(node.type) && node.type !== 'labelSuggester') {
    return false;
  }

  const data = node.data as Record<string, unknown>;
  if (data.sessionId || data.vlmResult) {
    return false;
  }

  const upstreamIds = edges
    .filter((edge) => edge.target === node.id)
    .map((edge) => edge.source);

  return !upstreamIds.some((upstreamId) => {
    const upstreamData = nodes.find((candidate) => candidate.id === upstreamId)?.data as
      | Record<string, unknown>
      | undefined;
    return Boolean(upstreamData?.sessionId || upstreamData?.vlmResult);
  });
}

function getString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function getNodeDataResult(node: Node<BaseNodeData> | undefined): NodeResult | undefined {
  if (!node) return undefined;

  const data = node.data as Record<string, unknown>;
  const filePath = getString(data.filePath) ||
    getString(data.path) ||
    getString(data.outputPath) ||
    getString(data.inputPath);

  const result: NodeResult = {
    sessionId: getString(data.sessionId),
    outputPath: getString(data.outputPath),
    filePath,
    fileType: data.fileType,
    volumeShape: data.volumeShape,
    metadata: data.metadata,
    configName: data.configName,
    annotations: data.annotations,
    sliceAnnotationsMap: data.sliceAnnotationsMap,
    sourcePath: data.sourcePath,
    segmentationResult: data.segmentationResult,
    segmentationShapes: data.segmentationShapes,
    segmentationPrompt: data.segmentationPrompt,
    annotationRecord: data.annotationRecord,
    annotationCount: data.annotationCount,
    vlmResult: data.vlmResult,
    labelSuggestions: data.labelSuggestions,
    labelSuggestionResult: data.labelSuggestionResult,
    voicePrompt: data.voicePrompt,
    transcript: data.transcript,
    audioPath: data.audioPath,
    campaign: data.campaign,
    campaignName: data.campaignName,
    assignmentResult: data.assignmentResult,
    assignmentCount: data.assignmentCount,
    sliceIndex: data.sliceIndex,
    view: data.view,
  };

  return Object.values(result).some((value) => value !== undefined)
    ? result
    : undefined;
}

function getUpstreamResults(
  nodeId: string,
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
  results: Map<string, NodeResult>,
) {
  const upstreamResults: NodeResult[] = [];

  for (const edge of edges) {
    if (edge.target !== nodeId) continue;

    const liveResult = results.get(edge.source);
    if (liveResult) {
      upstreamResults.push(liveResult);
      continue;
    }

    const sourceNode = nodes.find((node) => node.id === edge.source);
    const dataResult = getNodeDataResult(sourceNode);
    if (dataResult) {
      upstreamResults.push(dataResult);
    }
  }

  return upstreamResults;
}

/**
 * Detect the acquisition view plane from metadata.
 * Uses DICOM ImageOrientationPatient or NIfTI affine to determine whether the
 * volume was acquired axially, coronally, or sagittally. Falls back to axial.
 */
function detectViewPlane(
  metadata?: Record<string, unknown>,
): 'axial' | 'coronal' | 'sagittal' {
  if (!metadata) return 'axial';

  const iop = metadata.ImageOrientationPatient as number[] | undefined;
  if (iop && iop.length === 6) {
    const nx = Math.abs(iop[1] * iop[5] - iop[2] * iop[4]);
    const ny = Math.abs(iop[0] * iop[5] - iop[2] * iop[3]);
    const nz = Math.abs(iop[0] * iop[4] - iop[1] * iop[3]);

    if (nz >= nx && nz >= ny) return 'axial';
    if (ny >= nx && ny >= nz) return 'coronal';
    return 'sagittal';
  }

  const affine = metadata.affine as number[][] | undefined;
  if (affine && affine.length >= 3) {
    const absRow = [
      Math.abs(affine[0][2]),
      Math.abs(affine[1][2]),
      Math.abs(affine[2][2]),
    ];
    const maxIdx = absRow.indexOf(Math.max(...absRow));
    if (maxIdx === 2) return 'axial';
    if (maxIdx === 1) return 'coronal';
    return 'sagittal';
  }

  return 'axial';
}

function isView(value: unknown): value is 'axial' | 'coronal' | 'sagittal' {
  return value === 'axial' || value === 'coronal' || value === 'sagittal';
}

function shapesFromSegmentationResponse(
  shapes: api.SegmentationShape[],
): AnnotationShape[] {
  return shapes.map((shape) => ({
    type: 'polygon' as const,
    points: shape.points.map((point) => ({ x: point.x, y: point.y })),
    label: shape.label,
    color: shape.color,
  }));
}

function clampSliceIndex(index: number, totalSlices: number) {
  return Math.min(Math.max(0, index), Math.max(0, totalSlices - 1));
}

function getSegmentationSliceIndices(
  runMode: SegmentationRunMode,
  totalSlices: number,
  sliceIndex: number,
  sliceStart: number,
  sliceEnd: number,
  sliceStep: number,
) {
  const step = Math.max(1, Math.floor(sliceStep || 1));

  if (runMode === 'wholeVolume') {
    const indices: number[] = [];
    for (let index = 0; index < totalSlices; index += step) {
      indices.push(index);
    }
    return indices;
  }

  if (runMode === 'range') {
    const start = clampSliceIndex(Math.min(sliceStart, sliceEnd), totalSlices);
    const end = clampSliceIndex(Math.max(sliceStart, sliceEnd), totalSlices);
    const indices: number[] = [];
    for (let index = start; index <= end; index += step) {
      indices.push(index);
    }
    return indices;
  }

  return [clampSliceIndex(sliceIndex, totalSlices)];
}

function getStudyPath(data: Record<string, unknown>, upstreamResults: NodeResult[]) {
  const direct = getString(data.studyPath) || getString(data.sourcePath);
  if (direct) return direct;

  for (const upstream of upstreamResults) {
    const upstreamPath = getString(upstream.sourcePath) ||
      getString(upstream.filePath) ||
      getString(upstream.outputPath);
    if (upstreamPath) return upstreamPath;
  }

  return undefined;
}

function getAllSliceAnnotations(
  annotations: AnnotationShape[] | undefined,
  sliceAnnotationsMap: SliceAnnotationsMap | undefined,
  sliceIndex: number,
) {
  const map = { ...(sliceAnnotationsMap || {}) };
  if (annotations && annotations.length > 0) {
    map[sliceIndex] = annotations;
  }
  return map;
}

function annotationShapeToRecord(
  shape: AnnotationShape,
  sliceIndex: number,
  view: string,
  index: number,
) {
  return {
    annotation_id: `slice_${sliceIndex}_${view}_${index}`,
    slice_idx: sliceIndex,
    view_type: view,
    type: shape.type,
    label: shape.label,
    color: shape.color,
    shape,
  };
}

function flattenSliceAnnotations(
  annotations: AnnotationShape[] | undefined,
  sliceAnnotationsMap: SliceAnnotationsMap | undefined,
  sliceIndex: number,
  view: string,
) {
  const map = getAllSliceAnnotations(annotations, sliceAnnotationsMap, sliceIndex);
  return Object.entries(map).flatMap(([key, shapes]) => {
    const recordSliceIndex = Number(key);
    return (shapes || []).map((shape, index) =>
      annotationShapeToRecord(shape, recordSliceIndex, view, index),
    );
  });
}

function isAnnotationShape(value: unknown): value is AnnotationShape {
  return Boolean(value) &&
    typeof value === 'object' &&
    typeof (value as { type?: unknown }).type === 'string';
}

function recordToShape(record: Record<string, unknown>): AnnotationShape | null {
  if (isAnnotationShape(record.shape)) return record.shape;
  if (isAnnotationShape(record)) return record;
  return null;
}

function buildSliceAnnotationsMap(records: Array<Record<string, unknown>>) {
  const map: SliceAnnotationsMap = {};
  for (const record of records) {
    const shape = recordToShape(record);
    if (!shape) continue;
    const sliceIndex = Number(record.slice_idx ?? record.sliceIndex ?? 0);
    if (!map[sliceIndex]) map[sliceIndex] = [];
    map[sliceIndex].push(shape);
  }
  return map;
}

function getVlmModelForNode(type: string | undefined): VlmModelId {
  if (type === 'smolvlmNode') return 'smolvlm';
  if (type === 'medR1Node') return 'med-r1';
  return 'medgemma';
}

function isVlmModelId(value: unknown): value is VlmModelId {
  return value === 'medgemma' ||
    value === 'medgemma-1.5' ||
    value === 'medgemma-1.5-gguf' ||
    value === 'smolvlm' ||
    value === 'med-r1';
}

function isVlmModality(value: unknown): value is VlmModality {
  return value === 'MRI' || value === 'CT' || value === 'MG';
}

function detectVlmModality(
  metadata?: Record<string, unknown>,
  sourcePath?: string,
): VlmModality {
  const raw = [
    metadata?.Modality,
    metadata?.modality,
    metadata?.BodyPartExamined,
    metadata?.bodyPartExamined,
    metadata?.StudyDescription,
    metadata?.studyDescription,
    metadata?.SeriesDescription,
    metadata?.seriesDescription,
    metadata?.ProtocolName,
    metadata?.protocolName,
    sourcePath,
  ]
    .filter((value) => value !== undefined && value !== null)
    .map((value) => String(value))
    .join(' ')
    .toUpperCase();

  if (/\bCT\b/.test(raw) || raw.includes('COMPUTED TOMOGRAPHY') ||
    raw.includes('ABDOMEN') || raw.includes('ABDOMINAL')) return 'CT';
  if (/\bMG\b/.test(raw) || raw.includes('MAMMO') || raw.includes('MAMMOGRAPHY')) return 'MG';
  return 'MRI';
}

function resolveVlmModality(
  metadata?: Record<string, unknown>,
  sourcePath?: string,
  selected?: unknown,
): VlmModality {
  const detected = detectVlmModality(metadata, sourcePath);

  if (detected !== 'MRI') return detected;
  if (isVlmModality(selected)) return selected;
  return detected;
}

function normalizeVlmView(value: unknown): 'axial' | 'coronal' | 'sagittal' {
  return isView(value) ? value : 'axial';
}

function annotationLabels(
  annotations: AnnotationShape[] | undefined,
  sliceAnnotationsMap: SliceAnnotationsMap | undefined,
  sliceIndex: number,
) {
  const labels = new Set<string>();
  for (const shape of annotations || []) {
    if (shape.label?.trim()) labels.add(shape.label.trim());
  }
  for (const shape of sliceAnnotationsMap?.[sliceIndex] || []) {
    if (shape.label?.trim()) labels.add(shape.label.trim());
  }
  return [...labels];
}

function compactLabelCandidate(value: string) {
  return value
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<\/?answer>/gi, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[-*0-9.)\s]+/, '')
    .replace(/[.;:,]+$/, '')
    .trim();
}

function parseLabelCandidates(text: string, currentLabels: string[] = [], maxLabels = 12) {
  const current = new Set(currentLabels.map((label) => label.trim().toLowerCase()));
  const stopWords = new Set([
    'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an',
    'image', 'slice', 'label', 'labels', 'finding', 'findings',
  ]);
  const candidates: string[] = [];

  const regionMatches = [...text.matchAll(/(?:Region|ROI)\s*\d+\s*[:-]\s*([A-Za-z][A-Za-z0-9 /_-]{1,48})/gi)];
  candidates.push(...regionMatches.map((match) => match[1]));
  const bulletMatches = [...text.matchAll(/(?:^|\n)\s*(?:[-*]|\d+[.)])\s*([A-Za-z][A-Za-z0-9 /_-]{1,48})/g)];
  candidates.push(...bulletMatches.map((match) => match[1]));
  if (text.includes(',')) {
    candidates.push(...text.split(','));
  }

  const labels: string[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const label = compactLabelCandidate(candidate);
    const normalized = label.toLowerCase();
    if (!label || label.length < 2 || label.length > 40) continue;
    if (normalized.split(/\s+/).length > 5) continue;
    if (stopWords.has(normalized) || current.has(normalized) || seen.has(normalized)) continue;
    seen.add(normalized);
    labels.push(label);
    if (labels.length >= maxLabels) break;
  }

  return labels;
}

function labelSuggestionSliceKey(sliceIndex: number, view: 'axial' | 'coronal' | 'sagittal') {
  return `${view}:${sliceIndex}`;
}

function vlmSliceKey(sliceIndex: number, view: 'axial' | 'coronal' | 'sagittal') {
  return `${view}:${sliceIndex}`;
}

function normalizeVoiceSource(source: unknown): VoicePrompt['source'] {
  if (
    source === 'typed' ||
    source === 'dictation' ||
    source === 'upload' ||
    source === 'audio_path' ||
    source === 'manual'
  ) {
    return source;
  }
  return 'manual';
}

function isVoicePromptIntent(value: unknown): value is VoicePromptIntent {
  return value === 'describe' || value === 'anomaly' || value === 'both' || value === 'custom';
}

function voicePromptFromApi(res: api.VoicePromptResponse): VoicePrompt {
  return {
    text: res.transcript,
    intent: isVoicePromptIntent(res.intent) ? res.intent : 'both',
    promptKey: res.prompt_key || 'structured_radiology_review',
    identifyAnomalies: res.identify_anomalies,
    describeSlice: res.describe_slice,
    source: normalizeVoiceSource(res.source),
  };
}

function isVoicePrompt(value: unknown): value is VoicePrompt {
  return Boolean(value) &&
    typeof value === 'object' &&
    typeof (value as { text?: unknown }).text === 'string';
}

function getVoicePrompt(
  data: Record<string, unknown>,
  upstreamResults: NodeResult[],
): VoicePrompt | undefined {
  const upstreamVoice = upstreamResults
    .map((upstream) => upstream.voicePrompt)
    .find(isVoicePrompt);
  if (upstreamVoice) return upstreamVoice;

  if (isVoicePrompt(data.voicePrompt)) return data.voicePrompt;
  return undefined;
}

function mapCampaign(campaign: api.CollaborationCampaign): CampaignInfo {
  const assignments: ExpertAssignment[] = campaign.assignments.map((assignment) => ({
    expertId: assignment.expert_id,
    assignedPatients: assignment.assigned_patients,
    completedPatients: assignment.completed_patients,
    pendingPatients: assignment.pending_patients,
  }));

  return {
    name: campaign.name,
    datasetPath: campaign.dataset_path,
    description: campaign.description,
    createdAt: campaign.created_at,
    totalPatients: campaign.total_patients,
    patients: campaign.patients,
    progress: {
      totalPatients: campaign.progress.total_patients,
      assignedPatients: campaign.progress.assigned_patients,
      completed: campaign.progress.completed,
      reviewed: campaign.progress.reviewed,
      unassignedPatients: campaign.progress.unassigned_patients,
    },
    unassignedPatients: campaign.unassigned_patients,
    assignments,
  };
}

function isCampaignInfo(value: unknown): value is CampaignInfo {
  return Boolean(value) &&
    typeof value === 'object' &&
    typeof (value as { name?: unknown }).name === 'string';
}

function isPatientAssignmentResult(value: unknown): value is PatientAssignmentResult {
  return Boolean(value) &&
    typeof value === 'object' &&
    typeof (value as { campaignName?: unknown }).campaignName === 'string';
}

function getCampaignName(
  data: Record<string, unknown>,
  upstreamResults: NodeResult[],
): string | undefined {
  const direct = getString(data.campaignName);
  if (direct) return direct;

  if (isCampaignInfo(data.campaign)) return data.campaign.name;
  if (isPatientAssignmentResult(data.assignmentResult)) return data.assignmentResult.campaignName;

  for (const upstream of upstreamResults) {
    const upstreamName = getString(upstream.campaignName);
    if (upstreamName) return upstreamName;
    if (isCampaignInfo(upstream.campaign)) return upstream.campaign.name;
    if (isPatientAssignmentResult(upstream.assignmentResult)) {
      return upstream.assignmentResult.campaignName;
    }
  }

  return undefined;
}

function parsePatientIds(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item).trim())
      .filter(Boolean);
  }

  return String(value || '')
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function getImageContext(
  data: Record<string, unknown>,
  upstreamResults: NodeResult[],
  signal?: AbortSignal,
) {
  let sessionId = getString(data.sessionId);
  let sourcePath = getString(data.sourcePath) || getString(data.filePath);
  let metadata = data.metadata as Record<string, unknown> | undefined;
  let volumeShape = data.volumeShape as number[] | undefined;
  let sliceIndex = Number(data.sliceIndex ?? 0);
  let view = normalizeVlmView(data.view);
  let annotations = data.annotations as AnnotationShape[] | undefined;
  let sliceAnnotationsMap = data.sliceAnnotationsMap as SliceAnnotationsMap | undefined;

  for (const upstream of upstreamResults) {
    if (upstream.sessionId && !sessionId) {
      sessionId = upstream.sessionId;
    }
    if (!sourcePath) {
      sourcePath = getString(upstream.sourcePath) ||
        getString(upstream.filePath) ||
        getString(upstream.outputPath);
    }
    if (!metadata && upstream.metadata) {
      metadata = upstream.metadata as Record<string, unknown>;
    }
    if (!volumeShape && upstream.volumeShape) {
      volumeShape = upstream.volumeShape as number[];
    }
    if (upstream.sliceIndex !== undefined && data.sliceIndex === undefined) {
      sliceIndex = Number(upstream.sliceIndex ?? 0);
    }
    if (isView(upstream.view) && !isView(data.view)) {
      view = upstream.view;
    }
    if (!annotations && Array.isArray(upstream.annotations)) {
      annotations = upstream.annotations as AnnotationShape[];
    }
    if (!sliceAnnotationsMap && upstream.sliceAnnotationsMap) {
      sliceAnnotationsMap = upstream.sliceAnnotationsMap as SliceAnnotationsMap;
    }
    if (!sessionId && upstream.outputPath) {
      const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
      sessionId = loadRes.session_id;
      sourcePath = loadRes.file_path;
      metadata = loadRes.metadata;
      volumeShape = loadRes.volume_shape;
    }
  }

  return {
    sessionId,
    sourcePath,
    metadata,
    volumeShape,
    sliceIndex,
    view,
    annotations: annotations || [],
    sliceAnnotationsMap: sliceAnnotationsMap || {},
  };
}

/**
 * Execute the workflow graph.
 */
export async function executeWorkflow(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
  updateNodeData: (nodeId: string, data: Partial<BaseNodeData>) => void,
  options: ExecuteWorkflowOptions = {},
): Promise<WorkflowExecutionResult> {
  const mode = options.mode || 'all';
  const scopeIds = getScopedNodeIds(nodes, edges, options);
  const scopedNodes = nodes.filter((node) => scopeIds.has(node.id));
  const scopedEdges = edges.filter((edge) =>
    scopeIds.has(edge.source) && scopeIds.has(edge.target),
  );
  const pending = new Set(scopedNodes.map((node) => node.id));
  const results = new Map<string, NodeResult>();
  const executedNodeIds: string[] = [];
  const failedNodeIds: string[] = [];
  const skippedNodeIds: string[] = [];
  const failedOrSkipped = new Set<string>();
  const waitingNodeIds = new Set(
    scopedNodes
      .filter((node) => isWaitingExecutionNode(node, nodes, edges))
      .map((node) => node.id),
  );
  const waitingOrSkipped = new Set<string>();

  while (pending.size > 0) {
    assertNotAborted(options.signal);

    let skippedThisPass = false;
    for (const nodeId of [...pending]) {
      const blockedByFailedUpstream = scopedEdges.some((edge) =>
        edge.target === nodeId && failedOrSkipped.has(edge.source),
      );
      const blockedByWaitingUpstream = scopedEdges.some((edge) =>
        edge.target === nodeId && waitingOrSkipped.has(edge.source),
      );

      if (!blockedByFailedUpstream && !blockedByWaitingUpstream) continue;

      updateNodeData(nodeId, {
        status: 'skipped',
        error: blockedByWaitingUpstream
          ? 'Skipped because an upstream node is waiting for image review.'
          : 'Skipped because an upstream node failed.',
      });
      skippedNodeIds.push(nodeId);
      if (blockedByWaitingUpstream) {
        waitingOrSkipped.add(nodeId);
      } else {
        failedOrSkipped.add(nodeId);
      }
      pending.delete(nodeId);
      skippedThisPass = true;
    }

    const readyNodeIds = [...pending].filter((nodeId) =>
      scopedEdges
        .filter((edge) => edge.target === nodeId)
        .every((edge) => results.has(edge.source)),
    );

    if (readyNodeIds.length === 0) {
      if (skippedThisPass) continue;
      throw new Error('Cycle detected in workflow graph');
    }

    const outcomes = await Promise.all(readyNodeIds.map(async (nodeId) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return { nodeId, error: new Error('Node not found.') };

      const upstreamResults = getUpstreamResults(nodeId, nodes, edges, results);
      if (waitingNodeIds.has(nodeId)) {
        updateNodeData(nodeId, {
          status: 'waiting',
          error: undefined,
        });
        return { nodeId, waiting: true };
      }

      updateNodeData(nodeId, { status: 'running', error: undefined });

      try {
        assertNotAborted(options.signal);
        const result = await executeNode(node, upstreamResults, options.signal);
        assertNotAborted(options.signal);
        updateNodeData(nodeId, { status: 'success', ...result });
        return { nodeId, result };
      } catch (error) {
        if (isWorkflowExecutionCancelledError(error) || options.signal?.aborted) {
          updateNodeData(nodeId, {
            status: 'cancelled',
            error: 'Run cancelled.',
          });
          throw new WorkflowExecutionCancelledError();
        }

        const errorMsg = error instanceof Error ? error.message : String(error);
        updateNodeData(nodeId, { status: 'error', error: errorMsg });
        return {
          nodeId,
          error: new Error(`Node "${getNodeLabel(node)}" failed: ${errorMsg}`),
        };
      }
    })).catch((error) => {
      if (isWorkflowExecutionCancelledError(error) || options.signal?.aborted) {
        for (const nodeId of readyNodeIds) {
          updateNodeData(nodeId, {
            status: 'cancelled',
            error: 'Run cancelled.',
          });
        }
        throw new WorkflowExecutionCancelledError();
      }
      throw error;
    });

    for (const outcome of outcomes) {
      pending.delete(outcome.nodeId);

      if ('result' in outcome && outcome.result) {
        results.set(outcome.nodeId, outcome.result);
        executedNodeIds.push(outcome.nodeId);
        continue;
      }

      if ('waiting' in outcome && outcome.waiting) {
        skippedNodeIds.push(outcome.nodeId);
        waitingOrSkipped.add(outcome.nodeId);
        continue;
      }

      failedNodeIds.push(outcome.nodeId);
      failedOrSkipped.add(outcome.nodeId);
    }
  }

  if (failedNodeIds.length > 0) {
    const skippedSuffix = skippedNodeIds.length > 0
      ? ` ${skippedNodeIds.length} downstream node(s) skipped.`
      : '';
    throw new WorkflowExecutionFailedError(
      `${failedNodeIds.length} node(s) failed.${skippedSuffix}`,
      executedNodeIds,
      failedNodeIds,
      skippedNodeIds,
    );
  }

  return {
    mode,
    nodeIds: scopedNodes.map((node) => node.id),
    executedNodeIds,
    failedNodeIds,
    skippedNodeIds,
    results,
  };
}

/**
 * Execute a single node by calling the appropriate API endpoint.
 */
async function executeNode(
  node: Node<BaseNodeData>,
  upstreamResults: NodeResult[],
  signal?: AbortSignal,
): Promise<NodeResult> {
  const data = node.data;

  switch (node.type) {
    case 'dataLoader': {
      const path = (data as { path?: string }).path;
      if (!path) throw new Error('No path specified for DataLoaderNode');

      const res = await api.loadDataFromPath(path, signal);
      return {
        sessionId: res.session_id,
        fileType: res.file_type,
        volumeShape: res.volume_shape,
        metadata: res.metadata,
        filePath: res.file_path,
      };
    }

    case 'formatConverter': {
      const d = data as {
        inputPath?: string;
        conversionType?: string;
        outputPath?: string;
        axis?: number;
        compress?: boolean;
      };

      let inputPath = d.inputPath;
      if (!inputPath && upstreamResults.length > 0) {
        inputPath = upstreamResults[0].filePath as string | undefined;
        if (!inputPath) inputPath = upstreamResults[0].outputPath;
      }
      if (!inputPath) throw new Error('No input path for conversion');

      const convType = d.conversionType;
      if (!convType) throw new Error('No conversion type specified');

      const res = await api.runConversion(
        inputPath,
        convType,
        d.outputPath,
        {
          axis: d.axis,
          compress: d.compress,
        },
        signal,
      );
      return {
        outputPath: res.output_path,
        filePath: res.output_path,
        conversionType: res.conversion_type,
        outputSizeBytes: res.output_size_bytes,
      };
    }

    case 'metadataViewer': {
      const md = data as {
        sessionId?: string;
        sourcePath?: string;
      };

      let sessionId = md.sessionId;
      let sourcePath = md.sourcePath;

      for (const upstream of upstreamResults) {
        if (!sessionId && upstream.sessionId) {
          sessionId = upstream.sessionId;
        }
        if (!sourcePath) {
          sourcePath = upstream.filePath || upstream.outputPath;
        }
      }

      if (sessionId) {
        const metadataRes = await api.getMetadata(sessionId, signal);
        return {
          sessionId: metadataRes.session_id,
          filePath: metadataRes.file_path,
          fileType: metadataRes.file_type,
          volumeShape: metadataRes.volume_shape,
          metadata: metadataRes.metadata,
        };
      }

      if (sourcePath) {
        const loadRes = await api.loadDataFromPath(sourcePath, signal);
        return {
          sessionId: loadRes.session_id,
          filePath: loadRes.file_path,
          fileType: loadRes.file_type,
          volumeShape: loadRes.volume_shape,
          metadata: loadRes.metadata,
        };
      }

      throw new Error('No metadata source available - connect a Data Loader or Format Converter');
    }

    case 'deidentifyNode': {
      const dd = data as {
        sessionId?: string;
        sourcePath?: string;
        outputPath?: string;
        applyDeface?: boolean;
      };

      let sessionId = dd.sessionId;
      let sourcePath = dd.sourcePath;

      for (const upstream of upstreamResults) {
        if (!sessionId && upstream.sessionId) {
          sessionId = upstream.sessionId;
        }
        if (!sourcePath) {
          sourcePath = upstream.filePath || upstream.outputPath;
        }
      }

      if (!sessionId && !sourcePath) {
        throw new Error('Deidentify needs a loaded Data Loader session or source path.');
      }

      const res = await api.deidentifyData({
        session_id: sessionId,
        source_path: sourcePath,
        output_path: dd.outputPath,
        apply_deface: dd.applyDeface !== false,
      }, signal);

      return {
        sessionId: res.output_session_id,
        outputSessionId: res.output_session_id,
        sourceSessionId: res.session_id,
        sourcePath: res.source_path,
        outputPath: res.output_path,
        filePath: res.output_path,
        fileType: res.file_type,
        outputFileType: res.file_type,
        volumeShape: res.volume_shape,
        outputVolumeShape: res.volume_shape,
        metadata: res.sanitized_metadata,
        sanitizedMetadata: res.sanitized_metadata,
        sanitizedFields: res.sanitized_fields,
        sanitizedFieldCount: res.sanitized_field_count,
        filesProcessed: res.files_processed,
        auditPath: res.audit_path,
        message: res.message,
      };
    }

    case 'sliceViewer': {
      const sd = data as { sessionId?: string };

      let sessionId = sd.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (upstreamResults.length > 0) {
        const upstream = upstreamResults[0];
        if (upstream.sessionId && !sessionId) {
          sessionId = upstream.sessionId;
        }
        upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
        volumeShape = upstream.volumeShape as number[] | undefined;

        if (!sessionId && upstream.outputPath) {
          const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
          sessionId = loadRes.session_id;
          upstreamMeta = loadRes.metadata;
          volumeShape = loadRes.volume_shape;
        }
      }

      if (!sessionId) throw new Error('No data session available - connect a Data Loader or Format Converter');

      const detectedView = detectViewPlane(upstreamMeta);
      const sliceRes = await api.getSlice(sessionId, 0, detectedView, undefined, signal);
      return {
        sessionId,
        sliceIndex: sliceRes.slice_index,
        view: detectedView,
        totalSlices: sliceRes.total_slices,
        imageBase64: sliceRes.image_base64,
        volumeShape,
        metadata: upstreamMeta,
      };
    }

    case 'interactiveAnnotator': {
      const ad = data as {
        sessionId?: string;
        sliceIndex?: number;
        view?: string;
      };

      let sessionId = ad.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;
      let upstreamSegmentation: SegmentationResult | undefined;
      let upstreamPrompt: SegmentationPrompt | undefined;
      let segmentationShapes: AnnotationShape[] = [];
      let loadedAnnotations: AnnotationShape[] = [];
      let loadedAnnotationsMap: SliceAnnotationsMap | undefined;
      let labelSuggestions: string[] | undefined;
      let labelSuggestionResult: unknown;
      let sourcePath: string | undefined;
      let initialSliceIndex = Number(ad.sliceIndex ?? 0);
      let initialView: 'axial' | 'coronal' | 'sagittal' | undefined = isView(ad.view)
        ? ad.view
        : undefined;

      if (upstreamResults.length > 0) {
        for (const upstream of upstreamResults) {
          if (upstream.sessionId && !sessionId) {
            sessionId = upstream.sessionId;
            upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
            volumeShape = upstream.volumeShape as number[] | undefined;
            sourcePath = getString(upstream.sourcePath) || getString(upstream.filePath);
          }
          if (!sourcePath) {
            sourcePath = getString(upstream.sourcePath) ||
              getString(upstream.filePath) ||
              getString(upstream.outputPath);
          }
          if (upstream.segmentationPrompt && typeof upstream.segmentationPrompt === 'object') {
            upstreamPrompt = upstream.segmentationPrompt as SegmentationPrompt;
          }
          if (upstream.annotations || upstream.sliceAnnotationsMap) {
            loadedAnnotations = upstream.annotations as AnnotationShape[] || [];
            loadedAnnotationsMap = upstream.sliceAnnotationsMap as SliceAnnotationsMap | undefined;
            initialSliceIndex = Number(upstream.sliceIndex ?? initialSliceIndex);
            initialView = isView(upstream.view) ? upstream.view : initialView;
          }
          if (Array.isArray(upstream.labelSuggestions)) {
            labelSuggestions = upstream.labelSuggestions as string[];
            labelSuggestionResult = upstream.labelSuggestionResult;
          }
          if (upstream.segmentationResult && typeof upstream.segmentationResult === 'object') {
            upstreamSegmentation = upstream.segmentationResult as SegmentationResult;
            const firstSliceResult = upstreamSegmentation.sliceResults?.[0];
            segmentationShapes = firstSliceResult?.shapes || upstreamSegmentation.shapes || [];
            initialSliceIndex = firstSliceResult?.sliceIndex ?? upstreamSegmentation.sliceIndex ?? 0;
            initialView = upstreamSegmentation.view;
            if (!sessionId) sessionId = upstreamSegmentation.sessionId;
          } else if (Array.isArray(upstream.segmentationShapes)) {
            segmentationShapes = upstream.segmentationShapes as AnnotationShape[];
            initialSliceIndex = Number(upstream.sliceIndex ?? 0);
            initialView = isView(upstream.view) ? upstream.view : undefined;
          }
          if (!sessionId && upstream.outputPath) {
            const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
            sessionId = loadRes.session_id;
            upstreamMeta = loadRes.metadata;
            volumeShape = loadRes.volume_shape;
            sourcePath = loadRes.file_path;
          }
        }
      }

      if (!sessionId) throw new Error('No data session available - connect a Data Loader or Format Converter');

      const annotatorView = initialView || detectViewPlane(upstreamMeta);
      const annSliceRes = await api.getSlice(sessionId, initialSliceIndex, annotatorView, undefined, signal);
      const initialAnnotationsMap = upstreamSegmentation?.sliceResults?.length
        ? Object.fromEntries(upstreamSegmentation.sliceResults.map((sliceResult) => [
          sliceResult.sliceIndex,
          sliceResult.shapes,
        ]))
        : segmentationShapes.length > 0
          ? { [initialSliceIndex]: segmentationShapes }
          : loadedAnnotationsMap || (loadedAnnotations.length > 0
            ? { [initialSliceIndex]: loadedAnnotations }
            : {});
      const initialAnnotations = segmentationShapes.length > 0
        ? segmentationShapes
        : loadedAnnotations;
      return {
        sessionId,
        sliceIndex: annSliceRes.slice_index,
        view: annotatorView,
        totalSlices: annSliceRes.total_slices,
        imageBase64: annSliceRes.image_base64,
        annotations: initialAnnotations,
        sliceAnnotationsMap: initialAnnotationsMap,
        activeTool: 'pan',
        showLabels: true,
        sourcePath,
        volumeShape,
        metadata: upstreamMeta,
        modality: resolveVlmModality(upstreamMeta, sourcePath, data.modality),
        segmentationResult: upstreamSegmentation,
        segmentationPrompt: upstreamPrompt,
        labelSuggestions,
        labelSuggestionResult,
      };
    }

    case 'medsam2Segmenter': {
      const sd = data as {
        sessionId?: string;
        sliceIndex?: number;
        sliceStart?: number;
        sliceEnd?: number;
        sliceStep?: number;
        view?: string;
        configName?: string;
        runMode?: SegmentationRunMode;
        promptMode?: 'auto' | 'prompt';
        segmentationPrompt?: SegmentationPrompt;
      };

      let sessionId = sd.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;
      let configName = sd.configName || 'fast';
      let segmentationPrompt = sd.segmentationPrompt;

      for (const upstream of upstreamResults) {
        if (upstream.sessionId && !sessionId) {
          sessionId = upstream.sessionId;
          upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
          volumeShape = upstream.volumeShape as number[] | undefined;
        }
        if (upstream.configName && typeof upstream.configName === 'string') {
          configName = upstream.configName;
        }
        if (upstream.segmentationPrompt && typeof upstream.segmentationPrompt === 'object') {
          segmentationPrompt = upstream.segmentationPrompt as SegmentationPrompt;
        }
        if (!sessionId && upstream.outputPath) {
          const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
          sessionId = loadRes.session_id;
          upstreamMeta = loadRes.metadata;
          volumeShape = loadRes.volume_shape;
        }
      }

      if (!sessionId) throw new Error('No data session available - connect a Data Loader or Format Converter');

      const view = isView(sd.view) ? sd.view : detectViewPlane(upstreamMeta);
      const runMode = sd.runMode || 'single';
      if (segmentationPrompt || sd.promptMode === 'prompt') {
        if (!segmentationPrompt) {
          throw new Error('Prompt SAM2 needs a point or rectangle prompt from Interactive Annotator.');
        }

        const promptSessionId = segmentationPrompt.sessionId || sessionId;
        const promptView = segmentationPrompt.view || view;
        const promptSliceIndex = Number(segmentationPrompt.sliceIndex ?? sd.sliceIndex ?? 0);
        const res = await api.promptSegment(
          promptSessionId,
          promptSliceIndex,
          promptView,
          segmentationPrompt.points,
          segmentationPrompt.boxes,
          configName,
          signal,
        );
        const shapes = shapesFromSegmentationResponse(res.shapes);
        const sliceResults: SegmentationSliceResult[] = [{
          sliceIndex: promptSliceIndex,
          view: promptView,
          shapes,
          count: res.count,
          rawMaskCount: res.raw_mask_count,
          filteredCount: Math.max(0, res.raw_mask_count - res.count),
          elapsedSeconds: res.elapsed_seconds,
        }];
        const segmentationResult: SegmentationResult = {
          sessionId: promptSessionId,
          runMode: 'single',
          sliceIndex: promptSliceIndex,
          view: promptView,
          shapes,
          sliceResults,
          segmentedSliceCount: 1,
          count: res.count,
          rawMaskCount: res.raw_mask_count,
          filteredCount: Math.max(0, res.raw_mask_count - res.count),
          configUsed: configName,
          elapsedSeconds: res.elapsed_seconds,
          message: res.message,
        };

        return {
          sessionId: promptSessionId,
          runMode: 'single',
          promptMode: 'prompt',
          sliceIndex: promptSliceIndex,
          view: promptView,
          configName,
          segmentationPrompt,
          segmentationResult,
          segmentationShapes: shapes,
          segmentationSliceResults: sliceResults,
          segmentedSliceCount: 1,
          rawMaskCount: res.raw_mask_count,
          filteredCount: segmentationResult.filteredCount,
          elapsedSeconds: res.elapsed_seconds,
          metadata: upstreamMeta,
          volumeShape,
        };
      }
      const probeSlice = await api.getSlice(sessionId, 0, view, undefined, signal);
      const sliceIndices = getSegmentationSliceIndices(
        runMode,
        probeSlice.total_slices,
        Number(sd.sliceIndex ?? 0),
        Number(sd.sliceStart ?? 0),
        Number(sd.sliceEnd ?? 0),
        Number(sd.sliceStep ?? 1),
      );
      const sliceResults: SegmentationSliceResult[] = [];

      for (const sliceIndex of sliceIndices) {
        assertNotAborted(signal);
        const res = await api.autoSegment(sessionId, sliceIndex, view, configName, signal);
        sliceResults.push({
          sliceIndex,
          view,
          shapes: shapesFromSegmentationResponse(res.shapes),
          count: res.count,
          rawMaskCount: res.raw_mask_count,
          filteredCount: Math.max(0, res.raw_mask_count - res.count),
          elapsedSeconds: res.elapsed_seconds,
        });
      }

      const primaryResult = sliceResults[0];
      const shapes = primaryResult?.shapes || [];
      const totalCount = sliceResults.reduce((sum, result) => sum + result.count, 0);
      const totalRawMaskCount = sliceResults.reduce((sum, result) => sum + result.rawMaskCount, 0);
      const totalFilteredCount = sliceResults.reduce((sum, result) => sum + result.filteredCount, 0);
      const totalElapsedSeconds = sliceResults.reduce((sum, result) => sum + result.elapsedSeconds, 0);
      const segmentationResult: SegmentationResult = {
        sessionId,
        runMode,
        sliceIndex: primaryResult?.sliceIndex ?? 0,
        view,
        shapes,
        sliceResults,
        segmentedSliceCount: sliceResults.length,
        count: totalCount,
        rawMaskCount: totalRawMaskCount,
        filteredCount: totalFilteredCount,
        configUsed: configName,
        elapsedSeconds: totalElapsedSeconds,
        message: `SAM2 completed ${sliceResults.length} slice(s) in ${totalElapsedSeconds.toFixed(1)}s.`,
      };

      return {
        sessionId,
        runMode,
        sliceIndex: segmentationResult.sliceIndex,
        view,
        configName,
        segmentationResult,
        segmentationShapes: shapes,
        segmentationSliceResults: sliceResults,
        segmentedSliceCount: sliceResults.length,
        rawMaskCount: totalRawMaskCount,
        filteredCount: segmentationResult.filteredCount,
        elapsedSeconds: totalElapsedSeconds,
        metadata: upstreamMeta,
        volumeShape,
      };
    }

    case 'voiceInput': {
      const vd = data as {
        transcript?: string;
        audioPath?: string;
        intent?: VoicePromptIntent;
        promptKey?: string;
        autoDetectIntent?: boolean;
      };

      let voicePrompt: VoicePrompt | undefined;
      const transcript = getString(vd.transcript);

      if (transcript) {
        if (vd.autoDetectIntent === false) {
          const intent = isVoicePromptIntent(vd.intent) ? vd.intent : 'custom';
          const promptKey = vd.promptKey || 'describe_slice';
          voicePrompt = {
            text: transcript,
            intent,
            promptKey,
            identifyAnomalies: intent === 'anomaly' || intent === 'both',
            describeSlice: intent === 'describe' || intent === 'both',
            source: 'typed',
          };
        } else {
          const res = await api.inferVoiceIntent(transcript, signal);
          voicePrompt = voicePromptFromApi(res);
        }
      } else if (getString(vd.audioPath)) {
        const res = await api.transcribeVoicePath(String(vd.audioPath), signal);
        voicePrompt = voicePromptFromApi(res);
      }

      if (!voicePrompt) {
        throw new Error('Voice Input needs a transcript or audio file path.');
      }

      return {
        transcript: voicePrompt.text,
        intent: voicePrompt.intent,
        promptKey: voicePrompt.promptKey,
        customPrompt: voicePrompt.text,
        voicePrompt,
      };
    }

    case 'medgemmaNode':
    case 'smolvlmNode':
    case 'medR1Node': {
      const vd = data as Record<string, unknown>;
      const context = await getImageContext(vd, upstreamResults, signal);
      if (!context.sessionId) {
        throw new Error('VLM analysis needs a loaded image session from Data Loader, Format Converter, or Interactive Annotator.');
      }

      const model = isVlmModelId(vd.model) ? vd.model : getVlmModelForNode(node.type);
      const modality = resolveVlmModality(context.metadata, context.sourcePath, vd.modality);
      const voicePrompt = getVoicePrompt(vd, upstreamResults);
      const res = await api.runVlmAnalysis({
        session_id: context.sessionId,
        slice_index: context.sliceIndex,
        view: context.view,
        model,
        modality,
        prompt_key: voicePrompt?.promptKey || String(vd.promptKey || 'describe_slice'),
        custom_prompt: voicePrompt?.text || getString(vd.customPrompt),
        max_tokens: Number(vd.maxTokens || (model === 'medgemma-1.5-gguf' ? 512 : model === 'smolvlm' ? 128 : 256)),
        include_reasoning: Boolean(vd.includeReasoning),
        annotations: context.annotations,
        use_overlay: Boolean(vd.useOverlay),
      }, signal);

      const vlmResult: VlmAnalysisResult = {
        model,
        modelLabel: res.model_label,
        modality: res.modality,
        promptKey: res.prompt_key,
        promptTitle: res.prompt_title,
        promptUsed: res.prompt_used,
        sessionId: res.session_id,
        sliceIndex: res.slice_index,
        view: normalizeVlmView(res.view),
        text: res.text,
        labels: res.labels || [],
        elapsedSeconds: res.elapsed_seconds,
      };
      const sliceKey = vlmSliceKey(vlmResult.sliceIndex, vlmResult.view);

      return {
        sessionId: context.sessionId,
        sourcePath: context.sourcePath,
        sliceIndex: context.sliceIndex,
        view: context.view,
        modality,
        annotations: context.annotations,
        sliceAnnotationsMap: context.sliceAnnotationsMap,
        metadata: context.metadata,
        volumeShape: context.volumeShape,
        voicePrompt,
        vlmResult,
        vlmResultsBySlice: {
          ...(vd.vlmResultsBySlice as Record<string, VlmAnalysisResult> | undefined),
          [sliceKey]: vlmResult,
        },
      };
    }

    case 'labelSuggester': {
      const ld = data as Record<string, unknown>;
      const upstreamVlmResult = upstreamResults
        .map((upstream) => upstream.vlmResult)
        .find((result): result is VlmAnalysisResult => Boolean(result));
      const voicePrompt = getVoicePrompt(ld, upstreamResults);
      const model = isVlmModelId(ld.model)
        ? ld.model as VlmModelId
        : 'medgemma-1.5-gguf';
      const maxTokens = Number(ld.maxTokens || (model === 'medgemma-1.5-gguf' ? 512 : model === 'smolvlm' ? 128 : 256));
      const maxLabels = Number(ld.maxLabels || 12);
      const context = await getImageContext(ld, upstreamResults, signal);
      const modality = upstreamVlmResult?.modality ||
        resolveVlmModality(context.metadata, context.sourcePath, ld.modality);
      const currentLabels = annotationLabels(
        context.annotations,
        context.sliceAnnotationsMap,
        context.sliceIndex,
      );

      if (upstreamVlmResult && !context.sessionId) {
        const labelSuggestions = parseLabelCandidates(
          upstreamVlmResult.text,
          currentLabels,
          maxLabels,
        );
        const suggestionContext = {
          sliceIndex: Number(upstreamVlmResult.sliceIndex ?? 0),
          view: normalizeVlmView(upstreamVlmResult.view),
          sessionId: context.sessionId,
          sourcePath: context.sourcePath,
        };
        const sliceKey = labelSuggestionSliceKey(suggestionContext.sliceIndex, suggestionContext.view);
        const labelSuggestionResult = {
          model: upstreamVlmResult.model,
          modality: upstreamVlmResult.modality,
          labels: labelSuggestions,
          rawText: upstreamVlmResult.text,
          promptKey: upstreamVlmResult.promptKey,
          promptTitle: upstreamVlmResult.promptTitle,
          elapsedSeconds: upstreamVlmResult.elapsedSeconds,
        };
        return {
          labelSuggestions,
          currentLabels,
          labelSuggestionContext: suggestionContext,
          labelSuggestionResult,
          labelSuggestionsBySlice: {
            ...(ld.labelSuggestionsBySlice as Record<string, string[]> | undefined),
            [sliceKey]: labelSuggestions,
          },
          labelSuggestionResultsBySlice: {
            ...(ld.labelSuggestionResultsBySlice as Record<string, LabelSuggestionResult> | undefined),
            [sliceKey]: labelSuggestionResult,
          },
          vlmResult: upstreamVlmResult,
        };
      }

      if (!context.sessionId) {
        throw new Error('Label Suggester needs a loaded image session or an upstream VLM analysis.');
      }

      const res = await api.suggestVlmLabels({
        session_id: context.sessionId,
        slice_index: context.sliceIndex,
        view: context.view,
        model,
        modality,
        prompt_key: voicePrompt?.promptKey || String(ld.promptKey || 'annotation_label_candidates'),
        custom_prompt: voicePrompt?.text || getString(ld.customPrompt),
        max_tokens: maxTokens,
        current_labels: currentLabels,
        max_labels: maxLabels,
        annotations: context.annotations,
        use_overlay: ld.useOverlay !== false,
      }, signal);

      const vlmResult: VlmAnalysisResult = {
        model,
        modelLabel: res.model_label,
        modality: res.modality,
        promptKey: res.prompt_key,
        promptTitle: res.prompt_title,
        promptUsed: res.prompt_used,
        sessionId: res.session_id,
        sliceIndex: res.slice_index,
        view: normalizeVlmView(res.view),
        text: res.text,
        labels: res.labels || [],
        elapsedSeconds: res.elapsed_seconds,
      };

      const suggestionContext = {
        sliceIndex: context.sliceIndex,
        view: context.view,
        sessionId: context.sessionId,
        sourcePath: context.sourcePath,
      };
      const sliceKey = labelSuggestionSliceKey(suggestionContext.sliceIndex, suggestionContext.view);
      const labelSuggestionResult = {
        model,
        modality,
        labels: res.labels || [],
        rawText: res.text,
        promptKey: res.prompt_key,
        promptTitle: res.prompt_title,
        elapsedSeconds: res.elapsed_seconds,
      };

      return {
        sessionId: context.sessionId,
        sourcePath: context.sourcePath,
        sliceIndex: context.sliceIndex,
        view: context.view,
        model,
        modality,
        annotations: context.annotations,
        sliceAnnotationsMap: context.sliceAnnotationsMap,
        currentLabels,
        labelSuggestions: res.labels || [],
        voicePrompt,
        labelSuggestionContext: suggestionContext,
        labelSuggestionResult,
        labelSuggestionsBySlice: {
          ...(ld.labelSuggestionsBySlice as Record<string, string[]> | undefined),
          [sliceKey]: res.labels || [],
        },
        labelSuggestionResultsBySlice: {
          ...(ld.labelSuggestionResultsBySlice as Record<string, LabelSuggestionResult> | undefined),
          [sliceKey]: labelSuggestionResult,
        },
        vlmResult,
      };
    }

    case 'annotationStore': {
      const sd = data as {
        userId?: string;
        studyPath?: string;
        annotationType?: string;
        annotations?: AnnotationShape[];
        sliceAnnotationsMap?: SliceAnnotationsMap;
        sliceIndex?: number;
        view?: string;
      };
      const userId = sd.userId || 'workflow_user';
      const studyPath = getStudyPath(sd as Record<string, unknown>, upstreamResults);
      if (!studyPath) throw new Error('Annotation Store needs a study path.');

      const upstreamAnnotated = upstreamResults.find((upstream) =>
        upstream.annotations || upstream.sliceAnnotationsMap,
      );
      const annotations = (upstreamAnnotated?.annotations as AnnotationShape[] | undefined) ||
        sd.annotations ||
        [];
      const sliceAnnotationsMap = (upstreamAnnotated?.sliceAnnotationsMap as SliceAnnotationsMap | undefined) ||
        sd.sliceAnnotationsMap ||
        {};
      const sliceIndex = Number(upstreamAnnotated?.sliceIndex ?? sd.sliceIndex ?? 0);
      const view = String(upstreamAnnotated?.view || sd.view || 'axial');
      const sliceRecords = flattenSliceAnnotations(
        annotations,
        sliceAnnotationsMap,
        sliceIndex,
        view,
      );

      const res = await api.storeAnnotations({
        user_id: userId,
        study_path: studyPath,
        annotation_type: sd.annotationType || 'manual',
        slice_annotations: sliceRecords,
        study_metadata: {
          sessionId: upstreamAnnotated?.sessionId,
          view,
        },
      }, signal);
      const annotationRecord: AnnotationRecord = {
        userId: res.user_id,
        studyPath: res.study_path,
        annotationCount: res.annotation_count,
        message: res.message,
      };

      return {
        annotationRecord,
        annotationCount: res.annotation_count,
        userId,
        studyPath,
      };
    }

    case 'annotationLoad': {
      const ld = data as {
        userId?: string;
        studyPath?: string;
      };
      const userId = ld.userId || 'workflow_user';
      const studyPath = getStudyPath(ld as Record<string, unknown>, upstreamResults);
      if (!studyPath) throw new Error('Annotation Load needs a study path.');

      const res = await api.loadAnnotations(userId, studyPath, signal);
      const records = Array.isArray(res.data?.slice_annotations)
        ? res.data.slice_annotations as Array<Record<string, unknown>>
        : [];
      const sliceAnnotationsMap = buildSliceAnnotationsMap(records);
      const firstSliceKey = Object.keys(sliceAnnotationsMap)[0];
      const firstSliceIndex = Number(firstSliceKey ?? 0);
      const annotations = sliceAnnotationsMap[firstSliceIndex] || [];
      const annotationRecord: AnnotationRecord = {
        userId: res.user_id,
        studyPath: res.study_path,
        annotationCount: res.annotation_count,
        data: res.data || undefined,
        message: res.message,
      };

      return {
        sessionId: upstreamResults.find((upstream) => upstream.sessionId)?.sessionId,
        sourcePath: studyPath,
        studyPath,
        userId,
        annotationRecord,
        annotationCount: res.annotation_count,
        annotations,
        sliceAnnotationsMap,
        sliceIndex: firstSliceIndex,
        view: 'axial',
      };
    }

    case 'exportNode': {
      const ed = data as {
        userId?: string;
        studyPath?: string;
        exportFormat?: string;
        outputPath?: string;
        annotations?: AnnotationShape[];
        sliceAnnotationsMap?: SliceAnnotationsMap;
        sliceIndex?: number;
        view?: string;
      };
      const userId = ed.userId || 'workflow_user';
      const upstreamData = upstreamResults.find((upstream) =>
        Boolean(upstream.filePath || upstream.outputPath) && !upstream.annotationRecord,
      );
      if (upstreamData && !upstreamData.annotationRecord && !upstreamData.annotations && !upstreamData.sliceAnnotationsMap) {
        const outputPath = String(upstreamData.outputPath || upstreamData.filePath || '');
        return {
          outputPath,
          filePath: outputPath,
          sourcePath: upstreamData.sourcePath || outputPath,
          sessionId: upstreamData.sessionId,
          fileType: upstreamData.fileType,
          volumeShape: upstreamData.volumeShape,
          exportFormat: ed.exportFormat || 'data',
          message: 'Deidentified data is ready for export.',
        };
      }

      const studyPath = getStudyPath(ed as Record<string, unknown>, upstreamResults);
      if (!studyPath) throw new Error('Export needs a study path.');

      const upstreamRecord = upstreamResults
        .map((upstream) => upstream.annotationRecord)
        .find((record): record is AnnotationRecord => Boolean(record));
      const upstreamAnnotated = upstreamResults.find((upstream) =>
        upstream.annotations || upstream.sliceAnnotationsMap,
      );

      if (!upstreamRecord && upstreamAnnotated) {
        const records = flattenSliceAnnotations(
          upstreamAnnotated.annotations as AnnotationShape[] | undefined,
          upstreamAnnotated.sliceAnnotationsMap as SliceAnnotationsMap | undefined,
          Number(upstreamAnnotated.sliceIndex ?? ed.sliceIndex ?? 0),
          String(upstreamAnnotated.view || ed.view || 'axial'),
        );
        await api.storeAnnotations({
          user_id: userId,
          study_path: studyPath,
          annotation_type: 'manual',
          slice_annotations: records,
        }, signal);
      }

      const res = await api.exportAnnotations(
        userId,
        studyPath,
        ed.exportFormat || 'json',
        signal,
      );
      const annotationRecord: AnnotationRecord = {
        userId: res.user_id,
        studyPath: res.study_path,
        annotationCount: Number(upstreamRecord?.annotationCount ?? upstreamAnnotated?.annotationCount ?? 0),
        exportPath: res.export_path || undefined,
        message: res.message,
      };

      return {
        outputPath: res.export_path || undefined,
        filePath: res.export_path || undefined,
        exportFormat: res.output_format,
        annotationRecord,
        annotationCount: annotationRecord.annotationCount,
        userId,
        studyPath,
      };
    }

    case 'campaignSetup': {
      const cd = data as {
        campaignName?: string;
        datasetPath?: string;
        description?: string;
      };
      const campaignName = getString(cd.campaignName);
      const datasetPath = getString(cd.datasetPath);
      if (!campaignName) throw new Error('Campaign Setup needs a campaign name.');
      if (!datasetPath) throw new Error('Campaign Setup needs a dataset root path.');

      const res = await api.createCollaborationCampaign(
        campaignName,
        datasetPath,
        getString(cd.description) || '',
        signal,
      );
      const campaign = mapCampaign(res.campaign);

      return {
        campaign,
        campaignName: campaign.name,
        datasetPath: campaign.datasetPath,
        totalPatients: campaign.totalPatients,
        patients: campaign.patients,
        message: res.message,
      };
    }

    case 'patientAssign': {
      const pd = data as {
        campaignName?: string;
        expertId?: string;
        assignmentMode?: 'selected' | 'allUnassigned';
        patientIdsText?: string;
      };
      const campaignName = getCampaignName(pd as Record<string, unknown>, upstreamResults);
      const expertId = getString(pd.expertId);
      const assignmentMode = pd.assignmentMode || 'allUnassigned';
      if (!campaignName) throw new Error('Patient Assign needs a campaign from Campaign Setup or a campaign name.');
      if (!expertId) throw new Error('Patient Assign needs an expert ID.');

      const patientIds = assignmentMode === 'selected'
        ? parsePatientIds(pd.patientIdsText)
        : [];
      if (assignmentMode === 'selected' && patientIds.length === 0) {
        throw new Error('Selected assignment mode needs at least one patient ID.');
      }

      const res = await api.assignCollaborationPatients({
        campaign_name: campaignName,
        expert_id: expertId,
        patient_ids: patientIds,
        assignment_mode: assignmentMode,
      }, signal);
      const campaign = mapCampaign(res.campaign);
      const assignmentResult: PatientAssignmentResult = {
        campaignName: res.campaign_name,
        expertId: res.expert_id,
        assignedPatients: res.assigned_patients,
        assignmentCount: res.assignment_count,
        message: res.message,
      };

      return {
        campaign,
        campaignName: res.campaign_name,
        expertId: res.expert_id,
        assignedPatients: res.assigned_patients,
        assignmentCount: res.assignment_count,
        assignmentResult,
        message: res.message,
      };
    }

    case 'campaignStatus': {
      const sd = data as Record<string, unknown>;
      const campaignName = getCampaignName(sd, upstreamResults);
      if (!campaignName) throw new Error('Campaign Status needs a campaign input or campaign name.');

      const res = await api.getCollaborationCampaign(campaignName, signal);
      const campaign = mapCampaign(res);

      return {
        campaign,
        campaignName: campaign.name,
        totalPatients: campaign.totalPatients,
        assignedPatients: campaign.progress.assignedPatients,
        completed: campaign.progress.completed,
        reviewed: campaign.progress.reviewed,
      };
    }

    case 'crowdsourcingTasks': {
      return {
        userId: data.userId,
        currentPatientId: data.currentPatientId,
        tasks: data.tasks,
      };
    }

    case 'autoSegmentation': {
      const asd = data as { sessionId?: string; configName?: string };

      let sessionId = asd.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (!sessionId && upstreamResults.length > 0) {
        const upstream = upstreamResults[0];
        sessionId = upstream.sessionId;
        upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
        volumeShape = upstream.volumeShape as number[] | undefined;

        if (!sessionId && upstream.outputPath) {
          const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
          sessionId = loadRes.session_id;
          upstreamMeta = loadRes.metadata;
          volumeShape = loadRes.volume_shape;
        }
      }

      return {
        sessionId,
        configName: asd.configName || 'fast',
        metadata: upstreamMeta,
        volumeShape,
      };
    }

    default:
      throw new Error(`Unknown node type: ${node.type}`);
  }
}
