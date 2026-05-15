/**
 * Node Compatibility Engine
 * =========================
 * Validates connections from the centralized node contract registry.
 *
 * This file should stay thin: it answers whether two existing nodes may be
 * connected, while the actual compatibility rules live in nodeContracts.ts.
 * Keeping the rules contract-driven prevents quick-add, manual drag-connect,
 * and validation behavior from drifting apart.
 */

import type { Node, Edge, Connection } from '@xyflow/react';
import {
  formatNodeType,
  getAllowedSources,
  getAllowedTargets,
  getConnectionKinds,
} from './nodeContracts';

const REJECTION_REASONS: Record<string, string> = {
  selfLoop: 'A node cannot connect to itself',
  duplicate: 'These nodes are already connected',
  noOutput: 'This node type does not produce output',
};

export interface ValidationResult {
  valid: boolean;
  reason?: string;
}

function getSourceTarget(edgeOrConnection: Edge | Connection): {
  source: string | undefined;
  target: string | undefined;
} {
  return {
    source: edgeOrConnection.source ?? undefined,
    target: edgeOrConnection.target ?? undefined,
  };
}

export function validateConnection(
  connection: Edge | Connection,
  nodes: Node[],
  edges: Edge[],
): ValidationResult {
  const { source, target } = getSourceTarget(connection);

  if (source === target) {
    return { valid: false, reason: REJECTION_REASONS.selfLoop };
  }

  const alreadyExists = edges.some(
    (edge) => edge.source === source && edge.target === target,
  );
  if (alreadyExists) {
    return { valid: false, reason: REJECTION_REASONS.duplicate };
  }

  const sourceNode = nodes.find((node) => node.id === source);
  const targetNode = nodes.find((node) => node.id === target);
  if (!sourceNode || !targetNode) {
    return { valid: false, reason: 'Unknown node' };
  }

  const sourceType = sourceNode.type || '';
  const targetType = targetNode.type || '';
  const connectionKinds = getConnectionKinds(sourceType, targetType);

  if (connectionKinds.length === 0) {
    if (getAllowedTargets(sourceType).length === 0) {
      return { valid: false, reason: REJECTION_REASONS.noOutput };
    }

    return {
      valid: false,
      reason: `${formatNodeType(sourceType)} cannot connect to ${formatNodeType(targetType)}`,
    };
  }

  return { valid: true };
}

export function createIsValidConnection(
  nodes: Node[],
  edges: Edge[],
): (connection: Edge | Connection) => boolean {
  return (connection: Edge | Connection) =>
    validateConnection(connection, nodes, edges).valid;
}

export { getAllowedSources, getAllowedTargets };
