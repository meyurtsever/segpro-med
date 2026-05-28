/**
 * WorkflowLibrary - left sidebar for atomic nodes and end-to-end task templates.
 */

import { type DragEvent, useCallback, useState } from 'react';
import { nodePaletteItems } from '../nodes';
import { workflowTemplates, type WorkflowTemplate } from '../engine/workflowTemplates';

const EXPANDED_WIDTH = 260;
const COLLAPSED_WIDTH = 44;

const sidebarStyle: React.CSSProperties = {
  width: EXPANDED_WIDTH,
  flexShrink: 0,
  height: '100%',
  background: 'var(--bg-secondary)',
  borderRight: '1px solid var(--border-color)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  transition: 'width 0.18s ease',
};

const headerStyle: React.CSSProperties = {
  padding: '16px 16px 12px',
  borderBottom: '1px solid var(--border-color)',
  position: 'relative',
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
  transition: 'background 0.15s ease, border-color 0.15s ease',
  display: 'flex',
  gap: 10,
  alignItems: 'flex-start',
};

const categoryHeaderStyle: React.CSSProperties = {
  width: '100%',
  padding: '7px 8px',
  marginBottom: 4,
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  background: 'rgba(255, 255, 255, 0.03)',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: 7,
  textAlign: 'left',
};

const categoryChevronStyle: React.CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 12,
  fontWeight: 800,
  transition: 'transform 120ms ease',
};

const categoryCountStyle: React.CSSProperties = {
  marginLeft: 'auto',
  color: 'var(--text-muted)',
  fontSize: 10,
  fontWeight: 700,
};

const collapseButtonStyle: React.CSSProperties = {
  width: 28,
  height: 28,
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  fontSize: 13,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

interface NodePaletteProps {
  onApplyTemplate?: (template: WorkflowTemplate) => void;
}

export default function NodePalette({ onApplyTemplate }: NodePaletteProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<'nodes' | 'tasks'>('nodes');
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    'Data I/O': true,
  });

  const onDragStart = useCallback(
    (event: DragEvent, nodeType: string, defaultData: Record<string, unknown>) => {
      event.dataTransfer.setData('application/reactflow-type', nodeType);
      event.dataTransfer.setData('application/reactflow-data', JSON.stringify(defaultData));
      event.dataTransfer.effectAllowed = 'move';
    },
    [],
  );

  const categories = new Map<string, typeof nodePaletteItems>();
  for (const item of nodePaletteItems) {
    if (!categories.has(item.category)) {
      categories.set(item.category, []);
    }
    categories.get(item.category)!.push(item);
  }

  const toggleCategory = (category: string) => {
    setExpandedCategories((current) => ({
      ...current,
      [category]: !current[category],
    }));
  };

  if (collapsed) {
    return (
      <div style={{ ...sidebarStyle, width: COLLAPSED_WIDTH }}>
        <div
          style={{
            padding: 8,
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <button
            onClick={() => setCollapsed(false)}
            style={collapseButtonStyle}
            title="Expand workflow library"
            aria-label="Expand workflow library"
          >
            {'>'}
          </button>
        </div>
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '8px 0',
          }}
        >
          <div
            style={{
              transform: 'rotate(-90deg)',
              whiteSpace: 'nowrap',
              color: 'var(--text-secondary)',
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: 1,
              textTransform: 'uppercase',
            }}
          >
            Workflow Library
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={sidebarStyle}>
      <div style={headerStyle}>
        <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)' }}>
          Workflow Library
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
          Add nodes or complete task templates
        </div>
        <button
          onClick={() => setCollapsed(true)}
          style={{
            ...collapseButtonStyle,
            position: 'absolute',
            top: 14,
            right: 12,
          }}
          title="Collapse workflow library"
          aria-label="Collapse workflow library"
        >
          {'<'}
        </button>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 5,
            marginTop: 12,
          }}
        >
          {[
            ['nodes', 'Nodes'],
            ['tasks', 'End-to-end tasks'],
          ].map(([key, label]) => {
            const selected = activeTab === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => setActiveTab(key as 'nodes' | 'tasks')}
                style={{
                  padding: '7px 8px',
                  borderRadius: 6,
                  border: selected
                    ? '1px solid color-mix(in srgb, var(--accent-blue) 60%, var(--border-color))'
                    : '1px solid var(--border-color)',
                  background: selected
                    ? 'color-mix(in srgb, var(--accent-blue) 14%, var(--bg-tertiary))'
                    : 'var(--bg-tertiary)',
                  color: selected ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: 11,
                  fontWeight: 900,
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div style={listStyle}>
        {activeTab === 'tasks' ? (
          <div style={{ display: 'grid', gap: 8 }}>
            {workflowTemplates.map((template) => (
              <div
                key={template.id}
                style={{
                  padding: '10px 11px',
                  borderRadius: 8,
                  border: '1px solid var(--border-color)',
                  background: 'rgba(255, 255, 255, 0.03)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900 }}>
                    {template.title}
                  </div>
                  {template.automationLevel ? (
                    <span style={{
                      color: 'var(--accent-green)',
                      fontSize: 9,
                      fontWeight: 900,
                      textTransform: 'uppercase',
                      whiteSpace: 'nowrap',
                    }}>
                      Auto
                    </span>
                  ) : null}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, lineHeight: 1.35, marginTop: 4 }}>
                  {template.description}
                </div>
                <button
                  type="button"
                  onClick={() => onApplyTemplate?.(template)}
                  style={{
                    width: '100%',
                    marginTop: 9,
                    padding: '7px 9px',
                    borderRadius: 6,
                    border: '1px solid color-mix(in srgb, var(--accent-blue) 45%, var(--border-color))',
                    background: 'color-mix(in srgb, var(--accent-blue) 10%, var(--bg-tertiary))',
                    color: 'var(--accent-blue)',
                    cursor: 'pointer',
                    fontSize: 11,
                    fontWeight: 900,
                  }}
                  onMouseEnter={(event) => {
                    event.currentTarget.style.background = 'color-mix(in srgb, var(--accent-blue) 18%, var(--bg-tertiary))';
                  }}
                  onMouseLeave={(event) => {
                    event.currentTarget.style.background = 'color-mix(in srgb, var(--accent-blue) 10%, var(--bg-tertiary))';
                  }}
                >
                  Add task
                </button>
              </div>
            ))}
          </div>
        ) : Array.from(categories.entries()).map(([category, items]) => {
          const isExpanded = Boolean(expandedCategories[category]);

          return (
            <div key={category} style={{ marginBottom: 8 }}>
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                style={categoryHeaderStyle}
                aria-expanded={isExpanded}
                title={isExpanded ? `Collapse ${category}` : `Expand ${category}`}
                onMouseEnter={(event) => {
                  event.currentTarget.style.background = 'var(--bg-tertiary)';
                }}
                onMouseLeave={(event) => {
                  event.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                }}
              >
                <span
                  style={{
                    ...categoryChevronStyle,
                    transform: isExpanded ? 'rotate(90deg)' : 'none',
                  }}
                >
                  &gt;
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 800,
                    textTransform: 'uppercase',
                    letterSpacing: 0,
                    color: items[0].categoryColor,
                  }}
                >
                  {category}
                </span>
                <span style={categoryCountStyle}>{items.length}</span>
              </button>

              {isExpanded && items.map((item) => (
                <div
                  key={item.type}
                  draggable
                  onDragStart={(event) => onDragStart(event, item.type, item.defaultData)}
                  style={itemStyle}
                  onMouseEnter={(event) => {
                    event.currentTarget.style.background = 'var(--bg-tertiary)';
                    event.currentTarget.style.borderColor = 'var(--border-color)';
                  }}
                  onMouseLeave={(event) => {
                    event.currentTarget.style.background = 'transparent';
                    event.currentTarget.style.borderColor = 'transparent';
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
          );
        })}
      </div>
    </div>
  );
}
