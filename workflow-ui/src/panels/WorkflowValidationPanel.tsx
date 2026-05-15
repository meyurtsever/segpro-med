import type { CSSProperties } from 'react';
import type { WorkflowIssue } from '../engine/workflowValidation';

interface WorkflowValidationPanelProps {
  issues: WorkflowIssue[];
  onIssueClick: (issue: WorkflowIssue) => void;
  onClose: () => void;
}

const panelStyle: CSSProperties = {
  position: 'absolute',
  top: 56,
  right: 12,
  zIndex: 12,
  width: 330,
  borderRadius: 7,
  border: '1px solid color-mix(in srgb, var(--accent-orange) 35%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-orange) 8%, var(--bg-secondary))',
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const headerStyle: CSSProperties = {
  padding: '8px 10px',
  borderBottom: '1px solid var(--border-color)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
};

const titleStyle: CSSProperties = {
  color: 'var(--accent-orange)',
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

const issueButtonStyle = (severity: WorkflowIssue['severity']): CSSProperties => ({
  width: '100%',
  border: 'none',
  borderBottom: '1px solid rgba(255,255,255,0.06)',
  background: 'transparent',
  color: severity === 'error' ? 'var(--accent-red)' : 'var(--accent-orange)',
  cursor: 'pointer',
  padding: '8px 10px',
  textAlign: 'left',
  fontSize: 11,
  lineHeight: 1.35,
});

export default function WorkflowValidationPanel({
  issues,
  onIssueClick,
  onClose,
}: WorkflowValidationPanelProps) {
  const visibleIssues = issues.slice(0, 6);

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <div style={titleStyle}>Workflow checks</div>
        <button
          type="button"
          onClick={onClose}
          style={closeButtonStyle}
          title="Close workflow checks"
          aria-label="Close workflow checks"
        >
          x
        </button>
      </div>
      {visibleIssues.map((issue) => (
        <button
          key={issue.id}
          type="button"
          onClick={() => onIssueClick(issue)}
          style={issueButtonStyle(issue.severity)}
          onMouseEnter={(event) => {
            event.currentTarget.style.background = 'var(--bg-tertiary)';
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.background = 'transparent';
          }}
        >
          <strong>{issue.severity === 'error' ? 'Required' : 'Suggestion'}:</strong>{' '}
          {issue.message}
        </button>
      ))}
      {issues.length > visibleIssues.length ? (
        <div style={{ padding: '7px 10px', color: 'var(--text-muted)', fontSize: 10 }}>
          {issues.length - visibleIssues.length} more checks hidden.
        </div>
      ) : null}
    </div>
  );
}
