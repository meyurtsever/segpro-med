/**
 * Workflow Execution Engine
 * =========================
 * Topologically sorts the node graph and executes nodes
 * in dependency order, passing outputs as inputs to connected nodes.
 */

import type { Node, Edge } from '@xyflow/react';
import type { BaseNodeData } from '../types/nodes';
import * as api from '../api/client';

/** Get topologically sorted node IDs (Kahn's algorithm) */
function topologicalSort(nodes: Node[], edges: Edge[]): string[] {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    adj.set(node.id, []);
  }

  for (const edge of edges) {
    adj.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    sorted.push(id);
    for (const neighbor of adj.get(id) || []) {
      inDegree.set(neighbor, (inDegree.get(neighbor) || 0) - 1);
      if (inDegree.get(neighbor) === 0) queue.push(neighbor);
    }
  }

  if (sorted.length !== nodes.length) {
    throw new Error('Cycle detected in workflow graph');
  }

  return sorted;
}

/** Results stored per node after execution */
export interface NodeResult {
  sessionId?: string;
  outputPath?: string;
  [key: string]: unknown;
}

/**
 * Detect the acquisition view plane from metadata.
 * Uses DICOM ImageOrientationPatient or NIfTI affine to determine
 * whether the volume was acquired axially, coronally, or sagittally.
 * Falls back to 'axial' if undetermined.
 */
function detectViewPlane(
  metadata?: Record<string, unknown>,
): 'axial' | 'coronal' | 'sagittal' {
  if (!metadata) return 'axial';

  // --- DICOM: ImageOrientationPatient [row_x, row_y, row_z, col_x, col_y, col_z]
  const iop = metadata.ImageOrientationPatient as number[] | undefined;
  if (iop && iop.length === 6) {
    // Cross product of row and column vectors gives slice normal
    const nx = Math.abs(iop[1] * iop[5] - iop[2] * iop[4]);
    const ny = Math.abs(iop[0] * iop[5] - iop[2] * iop[3]);
    const nz = Math.abs(iop[0] * iop[4] - iop[1] * iop[3]);

    if (nz >= nx && nz >= ny) return 'axial';      // normal ≈ Z → axial
    if (ny >= nx && ny >= nz) return 'coronal';     // normal ≈ Y → coronal
    return 'sagittal';                                // normal ≈ X → sagittal
  }

  // --- NIfTI: affine matrix (4×4 stored as nested arrays)
  const affine = metadata.affine as number[][] | undefined;
  if (affine && affine.length >= 3) {
    // The third row (index 2) of the affine gives the slice direction
    const absRow = [
      Math.abs(affine[0][2]),  // X component of slice direction
      Math.abs(affine[1][2]),  // Y component
      Math.abs(affine[2][2]),  // Z component
    ];
    const maxIdx = absRow.indexOf(Math.max(...absRow));
    if (maxIdx === 2) return 'axial';
    if (maxIdx === 1) return 'coronal';
    return 'sagittal';
  }

  return 'axial';
}

/**
 * Execute the entire workflow graph.
 * @param nodes - React Flow nodes
 * @param edges - React Flow edges
 * @param updateNodeData - callback to update node status/data in the store
 */
export async function executeWorkflow(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
  updateNodeData: (nodeId: string, data: Partial<BaseNodeData>) => void,
): Promise<Map<string, NodeResult>> {
  const results = new Map<string, NodeResult>();

  // Sort nodes in execution order
  const sortedIds = topologicalSort(nodes, edges);

  for (const nodeId of sortedIds) {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) continue;

    // Get upstream results (from connected source nodes)
    const upstreamResults: NodeResult[] = [];
    for (const edge of edges) {
      if (edge.target === nodeId) {
        const r = results.get(edge.source);
        if (r) upstreamResults.push(r);
      }
    }

    // Mark as running
    updateNodeData(nodeId, { status: 'running', error: undefined });

    try {
      const result = await executeNode(node, upstreamResults);
      results.set(nodeId, result);
      updateNodeData(nodeId, { status: 'success', ...result });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      updateNodeData(nodeId, { status: 'error', error: errorMsg });
      // Stop execution on error
      throw new Error(`Node "${node.data.label}" failed: ${errorMsg}`);
    }
  }

  return results;
}

/**
 * Execute a single node by calling the appropriate API endpoint.
 */
async function executeNode(
  node: Node<BaseNodeData>,
  upstreamResults: NodeResult[],
): Promise<NodeResult> {
  const data = node.data;

  switch (node.type) {
    case 'dataLoader': {
      const path = (data as { path?: string }).path;
      if (!path) throw new Error('No path specified for DataLoaderNode');

      const res = await api.loadDataFromPath(path);
      return {
        sessionId: res.session_id,
        fileType: res.file_type,
        volumeShape: res.volume_shape,
        metadata: res.metadata,
        filePath: res.file_path,
      };
    }

    case 'formatConverter': {
      const d = data as { inputPath?: string; conversionType?: string; outputPath?: string };

      // If inputPath is not set, try to get it from upstream
      let inputPath = d.inputPath;
      if (!inputPath && upstreamResults.length > 0) {
        inputPath = upstreamResults[0].filePath as string | undefined;
        if (!inputPath) inputPath = upstreamResults[0].outputPath;
      }
      if (!inputPath) throw new Error('No input path for conversion');

      const convType = d.conversionType;
      if (!convType) throw new Error('No conversion type specified');

      const res = await api.runConversion(inputPath, convType, d.outputPath);
      return {
        outputPath: res.output_path,
        conversionType: res.conversion_type,
        outputSizeBytes: res.output_size_bytes,
      };
    }

    case 'sliceViewer': {
      const sd = data as { sessionId?: string };

      // Try to get sessionId from node data first, then from upstream
      let sessionId = sd.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (!sessionId && upstreamResults.length > 0) {
        const upstream = upstreamResults[0];
        sessionId = upstream.sessionId;
        upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
        volumeShape = upstream.volumeShape as number[] | undefined;

        // If upstream is a FormatConverter, its output is a file path — load it first
        if (!sessionId && upstream.outputPath) {
          const loadRes = await api.loadDataFromPath(upstream.outputPath);
          sessionId = loadRes.session_id;
          upstreamMeta = loadRes.metadata;
          volumeShape = loadRes.volume_shape;
        }
      }

      if (!sessionId) throw new Error('No data session available — connect a Data Loader or Format Converter');

      // Auto-detect best view plane from metadata
      const detectedView = detectViewPlane(upstreamMeta);

      // Fetch the first slice to populate the viewer
      const sliceRes = await api.getSlice(sessionId, 0, detectedView);
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

      // Try to get sessionId from node data first, then from upstream
      let sessionId = ad.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (!sessionId && upstreamResults.length > 0) {
        // Look through all upstream results (may have data source + autoSegmentation)
        for (const upstream of upstreamResults) {
          if (upstream.sessionId && !sessionId) {
            sessionId = upstream.sessionId;
            upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
            volumeShape = upstream.volumeShape as number[] | undefined;
          }
          // If upstream is a FormatConverter, load from its output
          if (!sessionId && upstream.outputPath) {
            const loadRes = await api.loadDataFromPath(upstream.outputPath);
            sessionId = loadRes.session_id;
            upstreamMeta = loadRes.metadata;
            volumeShape = loadRes.volume_shape;
          }
        }
      }

      if (!sessionId) throw new Error('No data session available — connect a Data Loader or Format Converter');

      // Auto-detect best view plane from metadata
      const annotatorView = detectViewPlane(upstreamMeta);

      // Fetch the first slice to populate the annotator canvas
      const annSliceRes = await api.getSlice(sessionId, 0, annotatorView);
      return {
        sessionId,
        sliceIndex: annSliceRes.slice_index,
        view: annotatorView,
        totalSlices: annSliceRes.total_slices,
        imageBase64: annSliceRes.image_base64,
        annotations: [],
        sliceAnnotationsMap: {},
        activeTool: 'select',
        showLabels: true,
        volumeShape,
        metadata: upstreamMeta,
      };
    }

    case 'autoSegmentation': {
      const asd = data as { sessionId?: string; configName?: string };

      // Inherit sessionId from upstream
      let sessionId = asd.sessionId;
      let upstreamMeta: Record<string, unknown> | undefined;
      let volumeShape: number[] | undefined;

      if (!sessionId && upstreamResults.length > 0) {
        const upstream = upstreamResults[0];
        sessionId = upstream.sessionId;
        upstreamMeta = upstream.metadata as Record<string, unknown> | undefined;
        volumeShape = upstream.volumeShape as number[] | undefined;

        if (!sessionId && upstream.outputPath) {
          const loadRes = await api.loadDataFromPath(upstream.outputPath);
          sessionId = loadRes.session_id;
          upstreamMeta = loadRes.metadata;
          volumeShape = loadRes.volume_shape;
        }
      }

      if (!sessionId) throw new Error('No data session available — connect a Data Loader');

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
