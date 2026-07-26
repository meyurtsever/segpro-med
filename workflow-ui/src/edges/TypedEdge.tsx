/**
 * Typed Workflow Edge
 * ===================
 * Renders React Flow edges with the semantic data type carried by the connection.
 *
 * Edge color and label come from nodeContracts.ts, making graph connections
 * read as "session", "file path", "seg config", or "annotations" instead of
 * generic links.
 */

import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';

import { getPortDefinition, type WorkflowPortKind } from '../engine/nodeContracts';

interface TypedEdgeData {
  kind?: WorkflowPortKind;
  label?: string;
  color?: string;
}

export default function TypedEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  selected,
  data,
}: EdgeProps) {
  const edgeData = data as TypedEdgeData | undefined;
  const definition = getPortDefinition(edgeData?.kind);
  const color = edgeData?.color || definition?.color || 'var(--accent-blue)';
  const label = edgeData?.label || definition?.edgeLabel || 'data';

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: color,
          strokeWidth: selected ? 2.4 : 1.6,
          strokeDasharray: 'none',
          animation: 'none',
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan"
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: 'all',
            padding: '3px 7px',
            borderRadius: 4,
            border: `1px solid color-mix(in srgb, ${color} 55%, var(--border-color))`,
            background: `color-mix(in srgb, ${color} 16%, var(--bg-secondary))`,
            color,
            fontSize: 9,
            fontWeight: 800,
            lineHeight: 1.1,
            textTransform: 'uppercase',
            boxShadow: 'var(--shadow)',
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
