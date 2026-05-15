import type { CSSProperties } from 'react';

interface CanvasEmptyStateProps {
  onCreateDataLoader: () => void;
  onCreateStarterWorkflow: () => void;
}

const shellStyle: CSSProperties = {
  position: 'absolute',
  inset: 0,
  zIndex: 5,
  pointerEvents: 'none',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};

const panelStyle: CSSProperties = {
  width: 320,
  borderRadius: 8,
  border: '1px solid var(--border-color)',
  background: 'rgba(31, 32, 53, 0.9)',
  boxShadow: 'var(--shadow)',
  padding: 12,
  pointerEvents: 'all',
};

const titleStyle: CSSProperties = {
  color: 'var(--text-primary)',
  fontSize: 14,
  fontWeight: 800,
  marginBottom: 4,
};

const textStyle: CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 11,
  lineHeight: 1.45,
  marginBottom: 10,
};

const actionsStyle: CSSProperties = {
  display: 'flex',
  gap: 8,
};

const buttonStyle = (primary = false): CSSProperties => ({
  flex: 1,
  borderRadius: 6,
  border: `1px solid ${primary ? 'var(--accent-blue)' : 'var(--border-color)'}`,
  background: primary ? 'rgba(79, 141, 245, 0.16)' : 'var(--bg-tertiary)',
  color: primary ? 'var(--accent-blue)' : 'var(--text-secondary)',
  fontSize: 11,
  fontWeight: 800,
  cursor: 'pointer',
  padding: '8px 9px',
});

export default function CanvasEmptyState({
  onCreateDataLoader,
  onCreateStarterWorkflow,
}: CanvasEmptyStateProps) {
  return (
    <div style={shellStyle}>
      <div style={panelStyle}>
        <div style={titleStyle}>Start workflow</div>
        <div style={textStyle}>
          Create a data entry point, or place the basic segmentation path.
        </div>
        <div style={actionsStyle}>
          <button type="button" style={buttonStyle(true)} onClick={onCreateDataLoader}>
            Data Loader
          </button>
          <button type="button" style={buttonStyle()} onClick={onCreateStarterWorkflow}>
            Starter Graph
          </button>
        </div>
      </div>
    </div>
  );
}
