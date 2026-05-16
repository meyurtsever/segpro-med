import type { CSSProperties } from 'react';

import type { WorkflowTemplate } from '../engine/workflowTemplates';

interface WorkflowTemplatePanelProps {
  templates: WorkflowTemplate[];
  onApply: (template: WorkflowTemplate) => void;
  onClose: () => void;
}

const panelStyle: CSSProperties = {
  position: 'absolute',
  top: 56,
  right: 12,
  zIndex: 12,
  width: 360,
  borderRadius: 7,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const headerStyle: CSSProperties = {
  padding: '9px 10px',
  borderBottom: '1px solid var(--border-color)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
};

const titleStyle: CSSProperties = {
  color: 'var(--accent-blue)',
  fontSize: 10,
  fontWeight: 800,
  letterSpacing: 0.7,
  textTransform: 'uppercase',
};

const closeButtonStyle: CSSProperties = {
  width: 20,
  height: 20,
  borderRadius: 4,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  fontSize: 12,
  lineHeight: 1,
};

const templateButtonStyle: CSSProperties = {
  width: '100%',
  border: 'none',
  borderBottom: '1px solid rgba(255,255,255,0.06)',
  background: 'transparent',
  color: 'var(--text-primary)',
  cursor: 'pointer',
  padding: '10px',
  textAlign: 'left',
};

export default function WorkflowTemplatePanel({
  templates,
  onApply,
  onClose,
}: WorkflowTemplatePanelProps) {
  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <div>
          <div style={titleStyle}>Workflow templates</div>
          <div style={{ marginTop: 3, color: 'var(--text-muted)', fontSize: 10 }}>
            Add an editable pre-connected graph.
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          style={closeButtonStyle}
          title="Close templates"
          aria-label="Close templates"
        >
          x
        </button>
      </div>

      {templates.map((template) => (
        <button
          key={template.id}
          type="button"
          onClick={() => onApply(template)}
          style={templateButtonStyle}
          onMouseEnter={(event) => {
            event.currentTarget.style.background = 'var(--bg-tertiary)';
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.background = 'transparent';
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 800 }}>
            {template.title}
          </div>
          <div style={{ marginTop: 3, fontSize: 10, lineHeight: 1.35, color: 'var(--text-muted)' }}>
            {template.description}
          </div>
          <div style={{ marginTop: 6, fontSize: 9, fontWeight: 800, color: 'var(--accent-blue)' }}>
            {template.nodes.length} nodes / {template.connections.length} connections
          </div>
        </button>
      ))}
    </div>
  );
}
