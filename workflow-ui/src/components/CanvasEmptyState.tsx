import type { CSSProperties } from 'react';

interface CanvasEmptyStateProps {
  onCreateDataLoader: () => void;
  onOpenEndToEndTasks: () => void;
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

const buttonStyle = (tone: 'blue' | 'green' | 'neutral' = 'neutral'): CSSProperties => {
  const color = tone === 'blue'
    ? 'var(--accent-blue)'
    : tone === 'green'
      ? 'var(--accent-green)'
      : 'var(--text-secondary)';

  return {
  flex: 1,
  borderRadius: 6,
  border: `1px solid ${tone === 'neutral' ? 'var(--border-color)' : color}`,
  background: tone === 'neutral' ? 'var(--bg-tertiary)' : `color-mix(in srgb, ${color} 14%, var(--bg-tertiary))`,
  color,
  fontSize: 11,
  fontWeight: 800,
  cursor: 'pointer',
  padding: '8px 9px',
  };
};

export default function CanvasEmptyState({
  onCreateDataLoader,
  onOpenEndToEndTasks,
}: CanvasEmptyStateProps) {
  return (
    <div style={shellStyle}>
      <div style={panelStyle}>
        <div style={titleStyle}>Start workflow</div>
        <div style={textStyle}>
          Create a data entry point, or place the basic segmentation path.
        </div>
        <div style={actionsStyle}>
          <button type="button" style={buttonStyle('blue')} onClick={onCreateDataLoader}>
            Data Loader
          </button>
          <button type="button" style={buttonStyle('green')} onClick={onOpenEndToEndTasks}>
            End-to-end tasks
          </button>
        </div>
      </div>
    </div>
  );
}
