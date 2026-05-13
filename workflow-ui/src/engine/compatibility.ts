/**
 * Node Compatibility Engine
 * ==========================
 * Defines which node types can connect to which, and provides
 * validation functions used by React Flow's isValidConnection callback.
 *
 * Rules:
 *   DataLoader      → FormatConverter, SliceViewer
 *   FormatConverter  → FormatConverter, SliceViewer
 *   SliceViewer      → (nothing — terminal node, no output handle)
 *
 * Self-loops and duplicate edges are also rejected.
 */

import type { Node, Edge, Connection } from '@xyflow/react';

// ---------------------------------------------------------------------------
// Compatibility matrix
// ---------------------------------------------------------------------------

/**
 * For each source node type, the set of valid target node types it can connect to.
 * If a source type is not listed, it cannot connect to anything.
 */
const ALLOWED_TARGETS: Record<string, Set<string>> = {
  dataLoader: new Set(['formatConverter', 'sliceViewer', 'interactiveAnnotator', 'autoSegmentation']),
  formatConverter: new Set(['formatConverter', 'sliceViewer', 'interactiveAnnotator', 'autoSegmentation']),
  interactiveAnnotator: new Set(['sliceViewer']),
  autoSegmentation: new Set(['interactiveAnnotator']),
  // sliceViewer is terminal — no entry means no valid targets
};

/**
 * Human-readable reasons for connection rejection.
 */
const REJECTION_REASONS: Record<string, string> = {
  selfLoop: 'A node cannot connect to itself',
  duplicate: 'These nodes are already connected',
  incompatible: 'These node types cannot be connected',
  noOutput: 'This node type does not produce output',
  noInput: 'This node type does not accept input',
};

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export interface ValidationResult {
  valid: boolean;
  reason?: string;
}

/**
 * Extract source/target from either a Connection or an Edge.
 */
function getSourceTarget(edgeOrConnection: Edge | Connection): {
  source: string | undefined;
  target: string | undefined;
} {
  return {
    source: edgeOrConnection.source ?? undefined,
    target: edgeOrConnection.target ?? undefined,
  };
}

/**
 * Check whether a proposed connection is valid.
 *
 * @param connection  The React Flow connection/edge being attempted
 * @param nodes       Current graph nodes (to look up node types)
 * @param edges       Current graph edges (to check for duplicates)
 */
export function validateConnection(
  connection: Edge | Connection,
  nodes: Node[],
  edges: Edge[],
): ValidationResult {
  const { source, target } = getSourceTarget(connection);

  // 1. Self-loop check
  if (source === target) {
    return { valid: false, reason: REJECTION_REASONS.selfLoop };
  }

  // 2. Duplicate edge check
  const alreadyExists = edges.some(
    (e) => e.source === source && e.target === target,
  );
  if (alreadyExists) {
    return { valid: false, reason: REJECTION_REASONS.duplicate };
  }

  // 3. Look up source/target node types
  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);

  if (!sourceNode || !targetNode) {
    return { valid: false, reason: 'Unknown node' };
  }

  const sourceType = sourceNode.type || '';
  const targetType = targetNode.type || '';

  // 4. Source type must have allowed targets
  const allowed = ALLOWED_TARGETS[sourceType];
  if (!allowed) {
    return { valid: false, reason: REJECTION_REASONS.noOutput };
  }

  // 5. Target type must be in the allowed set
  if (!allowed.has(targetType)) {
    return {
      valid: false,
      reason: `${formatNodeType(sourceType)} cannot connect to ${formatNodeType(targetType)}`,
    };
  }

  return { valid: true };
}

/**
 * Convenience wrapper for React Flow's `isValidConnection` callback.
 * Returns a simple boolean suitable for the prop.
 */
export function createIsValidConnection(
  nodes: Node[],
  edges: Edge[],
): (connection: Edge | Connection) => boolean {
  return (connection: Edge | Connection) => {
    return validateConnection(connection, nodes, edges).valid;
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Pretty-print a node type string */
function formatNodeType(type: string): string {
  const labels: Record<string, string> = {
    dataLoader: 'Data Loader',
    formatConverter: 'Format Converter',
    sliceViewer: 'Slice Viewer',
    interactiveAnnotator: 'Interactive Annotator',
    autoSegmentation: 'Auto Segmentation',
  };
  return labels[type] || type;
}

/**
 * Get the list of node types that a given source can connect to.
 * Useful for palette hints / tooltips.
 */
export function getAllowedTargets(sourceType: string): string[] {
  const allowed = ALLOWED_TARGETS[sourceType];
  return allowed ? [...allowed] : [];
}

/**
 * Get the list of node types that can connect TO a given target.
 */
export function getAllowedSources(targetType: string): string[] {
  const sources: string[] = [];
  for (const [src, targets] of Object.entries(ALLOWED_TARGETS)) {
    if (targets.has(targetType)) sources.push(src);
  }
  return sources;
}
