/**
 * NodePalette — Left sidebar with draggable node types.
 * Drag a node from here onto the canvas to add it to the workflow.
 */

import { type DragEvent, useCallback } from 'react';
import { nodePaletteItems } from '../nodes';

const sidebarStyle: React.CSSProperties = {
  width: 260,
  height: '100%',
  background: 'var(--bg-secondary)',
  borderRight: '1px solid var(--border-color)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
};

const headerStyle: React.CSSProperties = {
  padding: '16px 16px 12px',
  borderBottom: '1px solid var(--border-color)',
};

const listStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '8px',
};

const itemStyle: React.CSSProperties = {
  padding: '10px 12px',
  marginBottom: 4,
  borderRadius: 8,
  border: '1px solid transparent',
  cursor: 'grab',
  transition: 'all 0.15s ease',
  display: 'flex',
  gap: 10,
  alignItems: 'flex-start',
};

export default function NodePalette() {
  const onDragStart = useCallback(
    (event: DragEvent, nodeType: string, defaultData: Record<string, unknown>) => {
      event.dataTransfer.setData('application/reactflow-type', nodeType);
      event.dataTransfer.setData('application/reactflow-data', JSON.stringify(defaultData));
      event.dataTransfer.effectAllowed = 'move';
    },
    [],
  );

  // Group items by category
  const categories = new Map<string, typeof nodePaletteItems>();
  for (const item of nodePaletteItems) {
    if (!categories.has(item.category)) {
      categories.set(item.category, []);
    }
    categories.get(item.category)!.push(item);
  }

  return (
    <div style={sidebarStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
          🧩 Node Palette
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
          Drag nodes onto the canvas
        </div>
      </div>

      {/* Node list */}
      <div style={listStyle}>
        {Array.from(categories.entries()).map(([category, items]) => (
          <div key={category} style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.8px',
                color: items[0].categoryColor,
                padding: '4px 8px',
                marginBottom: 4,
              }}
            >
              {category}
            </div>

            {items.map((item) => (
              <div
                key={item.type}
                draggable
                onDragStart={(e) => onDragStart(e, item.type, item.defaultData)}
                style={itemStyle}
                onMouseEnter={(e) => {
                  const el = e.currentTarget;
                  el.style.background = 'var(--bg-tertiary)';
                  el.style.borderColor = 'var(--border-color)';
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget;
                  el.style.background = 'transparent';
                  el.style.borderColor = 'transparent';
                }}
              >
                <span style={{ fontSize: 20, lineHeight: 1 }}>{item.icon}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    {item.description}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
