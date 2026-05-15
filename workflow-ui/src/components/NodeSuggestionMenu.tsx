import { ViewportPortal, type Node } from '@xyflow/react';
import type { CSSProperties } from 'react';

import type { BaseNodeData } from '../types/nodes';
import { nodePaletteItems } from '../engine/nodeContracts';

type PaletteItem = (typeof nodePaletteItems)[number];

interface EdgePreview {
  label: string;
  color: string;
}

interface NodeSuggestionMenuProps {
  direction: 'input' | 'output';
  position: { x: number; y: number };
  existingNodes: Node<BaseNodeData>[];
  suggestions: PaletteItem[];
  getEdgePreview: (candidateType: string | undefined) => EdgePreview | undefined;
  onClose: () => void;
  onConnectExisting: (nodeId: string) => void;
  onCreateNode: (nodeType: string, defaultData: Record<string, unknown>) => void;
}

const menuStyle: CSSProperties = {
  position: 'absolute',
  zIndex: 50,
  width: 226,
  pointerEvents: 'all',
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-color)',
  borderRadius: 7,
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const headerStyle: CSSProperties = {
  padding: '7px 9px',
  borderBottom: '1px solid var(--border-color)',
  color: 'var(--text-secondary)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0.7,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
};

const closeButtonStyle: CSSProperties = {
  width: 18,
  height: 18,
  borderRadius: 4,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  fontSize: 12,
  lineHeight: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

const itemStyle: CSSProperties = {
  width: '100%',
  border: 'none',
  background: 'transparent',
  color: 'var(--text-primary)',
  padding: '7px 9px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'flex-start',
  gap: 7,
  textAlign: 'left',
};

const sectionTitleStyle: CSSProperties = {
  padding: '7px 9px 3px',
  color: 'var(--text-muted)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0.7,
};

const dividerStyle: CSSProperties = {
  height: 1,
  background: 'var(--border-color)',
  margin: '4px 0',
};

const chipStyle = (color: string): CSSProperties => ({
  display: 'inline-flex',
  marginTop: 3,
  padding: '2px 5px',
  borderRadius: 4,
  border: `1px solid color-mix(in srgb, ${color} 52%, var(--border-color))`,
  background: `color-mix(in srgb, ${color} 13%, var(--bg-secondary))`,
  color,
  fontSize: 8,
  fontWeight: 800,
  lineHeight: 1.1,
  textTransform: 'uppercase',
});

function setHoverBackground(element: HTMLElement, hovered: boolean) {
  element.style.background = hovered ? 'var(--bg-tertiary)' : 'transparent';
}

export default function NodeSuggestionMenu({
  direction,
  position,
  existingNodes,
  suggestions,
  getEdgePreview,
  onClose,
  onConnectExisting,
  onCreateNode,
}: NodeSuggestionMenuProps) {
  const hasExisting = existingNodes.length > 0;
  const hasSuggestions = suggestions.length > 0;

  return (
    <ViewportPortal>
      <div
        style={{
          ...menuStyle,
          transform: `translate(${position.x}px, ${position.y}px)`,
        }}
        className="nodrag nopan nowheel"
        onPointerDown={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
      >
        <div style={headerStyle}>
          <span>
            {direction === 'output'
              ? 'Compatible next nodes'
              : 'Compatible previous nodes'}
          </span>
          <button
            type="button"
            onClick={onClose}
            style={closeButtonStyle}
            title="Close suggestions"
            aria-label="Close suggestions"
          >
            x
          </button>
        </div>

        {hasExisting ? (
          <>
            <div style={sectionTitleStyle}>Connect existing</div>
            {existingNodes.map((node) => {
              const item = nodePaletteItems.find((paletteNode) => paletteNode.type === node.type);
              const edgePreview = getEdgePreview(node.type);
              return (
                <button
                  key={node.id}
                  type="button"
                  onClick={() => onConnectExisting(node.id)}
                  style={itemStyle}
                  onMouseEnter={(event) => setHoverBackground(event.currentTarget, true)}
                  onMouseLeave={(event) => setHoverBackground(event.currentTarget, false)}
                >
                  <span style={{ fontSize: 15, lineHeight: 1 }}>{item?.icon || '*'}</span>
                  <span style={{ minWidth: 0 }}>
                    <span style={{ display: 'block', fontSize: 11, fontWeight: 700 }}>
                      {String(node.data.label || item?.label || node.type)}
                    </span>
                    {edgePreview ? (
                      <span style={chipStyle(edgePreview.color)}>
                        {edgePreview.label}
                      </span>
                    ) : null}
                  </span>
                </button>
              );
            })}
          </>
        ) : null}

        {hasExisting && hasSuggestions ? <div style={dividerStyle} /> : null}
        <div style={sectionTitleStyle}>Create new</div>
        {!hasSuggestions ? (
          <div style={{ padding: 10, color: 'var(--text-muted)', fontSize: 12 }}>
            No compatible node types.
          </div>
        ) : (
          suggestions.map((item) => {
            const edgePreview = getEdgePreview(item.type);
            return (
              <button
                key={item.type}
                type="button"
                onClick={() => onCreateNode(item.type, item.defaultData)}
                style={itemStyle}
                onMouseEnter={(event) => setHoverBackground(event.currentTarget, true)}
                onMouseLeave={(event) => setHoverBackground(event.currentTarget, false)}
              >
                <span style={{ fontSize: 15, lineHeight: 1 }}>{item.icon}</span>
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', fontSize: 11, fontWeight: 700 }}>
                    {item.label}
                  </span>
                  <span
                    style={{
                      display: 'block',
                      fontSize: 9,
                      color: 'var(--text-muted)',
                      marginTop: 1,
                      lineHeight: 1.35,
                    }}
                  >
                    {item.description}
                  </span>
                  {edgePreview ? (
                    <span style={chipStyle(edgePreview.color)}>
                      {edgePreview.label}
                    </span>
                  ) : null}
                </span>
              </button>
            );
          })
        )}
      </div>
    </ViewportPortal>
  );
}
