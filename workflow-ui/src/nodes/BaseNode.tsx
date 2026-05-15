/**
 * BaseNode — Shared chrome for all SegPro-Med workflow nodes.
 * Provides consistent title bar, status indicator, handles, and "Inspect on Tool" button.
 */

import { memo, useState, type MouseEvent, type ReactNode } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeStatus } from '../types/nodes';
import InfoModal, { type NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';

interface BaseNodeProps {
  nodeId?: string;
  nodeType?: string;
  title: string;
  icon: string;
  color: string;
  status: NodeStatus;
  error?: string;
  hasInput?: boolean;
  hasOutput?: boolean;
  inspectUrl?: string;
  info?: NodeInfo;
  children: ReactNode;
}

const statusConfig: Record<NodeStatus, { icon: string; color: string; label: string }> = {
  idle:    { icon: '⏸', color: 'var(--text-muted)',    label: 'Idle' },
  running: { icon: '⏳', color: 'var(--accent-orange)', label: 'Running' },
  success: { icon: '✅', color: 'var(--accent-green)',  label: 'Done' },
  error:   { icon: '❌', color: 'var(--accent-red)',    label: 'Error' },
};

function BaseNode({
  nodeId,
  nodeType,
  title,
  icon,
  color,
  status,
  error,
  hasInput = true,
  hasOutput = true,
  inspectUrl,
  info,
  children,
}: BaseNodeProps) {
  const s = statusConfig[status];
  const [showInfo, setShowInfo] = useState(false);
  const isSelected = useWorkflowStore((state) => (
    nodeId ? state.selectedNodeId === nodeId : false
  ));
  const borderColor = isSelected && color ? color : 'var(--border-color)';

  const handleHandleClick = (event: MouseEvent, direction: 'input' | 'output') => {
    if (!nodeId || !nodeType) return;
    event.preventDefault();
    event.stopPropagation();
    window.dispatchEvent(
      new CustomEvent('segpro:open-node-suggestions', {
        detail: {
          sourceNodeId: nodeId,
          sourceNodeType: nodeType,
          direction,
          clientX: event.clientX,
          clientY: event.clientY,
        },
      }),
    );
  };

  return (
    <div
      style={{
        background: 'var(--bg-node)',
        border: `1px solid ${borderColor}`,
        borderRadius: 10,
        minWidth: 280,
        maxWidth: 340,
        overflow: 'hidden',
        transition: 'border-color 120ms ease',
      }}
    >
      {/* Input handle */}
      {hasInput && (
        <Handle
          type="target"
          position={Position.Left}
          onClick={(event) => handleHandleClick(event, 'input')}
          title="Click to add a compatible previous node, or drag to connect manually"
          style={{ top: 24, cursor: 'pointer' }}
        />
      )}

      {/* Title bar */}
      <div
        style={{
          background: `${color}18`,
          borderBottom: `1px solid ${color}30`,
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 18 }}>{icon}</span>
        <span
          style={{
            flex: 1,
            fontWeight: 600,
            fontSize: 13,
            color: 'var(--text-primary)',
          }}
        >
          {title}
        </span>
        <span title={s.label} style={{ fontSize: 14, color: s.color }}>
          {s.icon}
        </span>
      </div>

      {/* Body */}
      <div style={{ padding: '10px 12px' }} className="nodrag nowheel">
        {children}
      </div>

      {/* Error message */}
      {error && (
        <div
          style={{
            padding: '6px 12px',
            fontSize: 11,
            color: 'var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.08)',
            borderTop: '1px solid rgba(224, 92, 92, 0.2)',
            wordBreak: 'break-word',
          }}
        >
          {error}
        </div>
      )}

      {/* Footer — Info & Inspect buttons */}
      {(info || inspectUrl) && (
        <div
          style={{
            borderTop: '1px solid var(--border-color)',
            padding: '6px 12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          {info ? (
            <button
              onClick={() => setShowInfo(true)}
              style={{
                background: 'transparent',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)',
                padding: '3px 10px',
                borderRadius: 4,
                fontSize: 11,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLElement).style.borderColor = 'var(--accent-purple)';
                (e.target as HTMLElement).style.color = 'var(--accent-purple)';
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLElement).style.borderColor = 'var(--border-color)';
                (e.target as HTMLElement).style.color = 'var(--text-secondary)';
              }}
            >
              ℹ️ About
            </button>
          ) : (
            <div />
          )}
          {inspectUrl ? (
            <button
              onClick={() => window.open(inspectUrl, '_blank')}
              style={{
                background: 'transparent',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)',
                padding: '3px 10px',
                borderRadius: 4,
                fontSize: 11,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLElement).style.borderColor = 'var(--accent-blue)';
                (e.target as HTMLElement).style.color = 'var(--accent-blue)';
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLElement).style.borderColor = 'var(--border-color)';
                (e.target as HTMLElement).style.color = 'var(--text-secondary)';
              }}
            >
              🔍 Inspect on Tool
            </button>
          ) : (
            <div />
          )}
        </div>
      )}

      {/* Info Modal */}
      {info && (
        <InfoModal
          isOpen={showInfo}
          onClose={() => setShowInfo(false)}
          title={title}
          icon={icon}
          color={color}
          info={info}
        />
      )}

      {/* Output handle */}
      {hasOutput && (
        <Handle
          type="source"
          position={Position.Right}
          onClick={(event) => handleHandleClick(event, 'output')}
          title="Click to add a compatible next node, or drag to connect manually"
          style={{ top: 24, cursor: 'pointer' }}
        />
      )}
    </div>
  );
}

export default memo(BaseNode);
