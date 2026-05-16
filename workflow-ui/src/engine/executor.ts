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
import type { BaseNodeData } from '../types/nodes';
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

  while (pending.size > 0) {
    assertNotAborted(options.signal);

    let skippedThisPass = false;
    for (const nodeId of [...pending]) {
      const blockedByFailedUpstream = scopedEdges.some((edge) =>
        edge.target === nodeId && failedOrSkipped.has(edge.source),
      );

      if (!blockedByFailedUpstream) continue;

      updateNodeData(nodeId, {
        status: 'skipped',
        error: 'Skipped because an upstream node failed.',
      });
      skippedNodeIds.push(nodeId);
      failedOrSkipped.add(nodeId);
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

    case 'sliceViewer': {
      const sd = data as { sessionId?: string };

      let sessionId = sd.sessionId;
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
      const ad = data as { sessionId?: string };

      let sessionId = ad.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (!sessionId && upstreamResults.length > 0) {
        for (const upstream of upstreamResults) {
          if (upstream.sessionId && !sessionId) {
            sessionId = upstream.sessionId;
            upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
            volumeShape = upstream.volumeShape as number[] | undefined;
          }
          if (!sessionId && upstream.outputPath) {
            const loadRes = await api.loadDataFromPath(upstream.outputPath, signal);
            sessionId = loadRes.session_id;
            upstreamMeta = loadRes.metadata;
            volumeShape = loadRes.volume_shape;
          }
        }
      }

      if (!sessionId) throw new Error('No data session available - connect a Data Loader or Format Converter');

      const annotatorView = detectViewPlane(upstreamMeta);
      const annSliceRes = await api.getSlice(sessionId, 0, annotatorView, undefined, signal);
      return {
        sessionId,
        sliceIndex: annSliceRes.slice_index,
        view: annotatorView,
        totalSlices: annSliceRes.total_slices,
        imageBase64: annSliceRes.image_base64,
        annotations: [],
        sliceAnnotationsMap: {},
        activeTool: 'rect',
        showLabels: true,
        volumeShape,
        metadata: upstreamMeta,
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

      if (!sessionId) throw new Error('No data session available - connect a Data Loader');

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
