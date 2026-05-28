import { memo } from 'react';
import { createPortal } from 'react-dom';

interface ResultModalProps {
  isOpen: boolean;
  title: string;
  status: 'success' | 'error';
  summary: string;
  outputPath?: string;
  expectation: string;
  nextStep: string;
  doNotShowAgain: boolean;
  onDoNotShowAgainChange: (checked: boolean) => void;
  onClose: () => void;
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 9999,
  background: 'rgba(0, 0, 0, 0.6)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 16,
  backdropFilter: 'blur(4px)',
};

const modalStyle: React.CSSProperties = {
  width: 480,
  maxWidth: '100%',
  maxHeight: '82vh',
  overflow: 'hidden',
  borderRadius: 12,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
  display: 'flex',
  flexDirection: 'column',
};

const buttonStyle = (color: string): React.CSSProperties => ({
  padding: '8px 14px',
  borderRadius: 6,
  border: `1px solid color-mix(in srgb, ${color} 54%, var(--border-color))`,
  background: `color-mix(in srgb, ${color} 15%, var(--bg-secondary))`,
  color,
  fontWeight: 800,
  fontSize: 12,
  cursor: 'pointer',
});

function ResultModal({
  isOpen,
  title,
  status,
  summary,
  outputPath,
  expectation,
  nextStep,
  doNotShowAgain,
  onDoNotShowAgainChange,
  onClose,
}: ResultModalProps) {
  if (!isOpen || typeof document === 'undefined') return null;
  const color = status === 'success' ? 'var(--accent-green)' : 'var(--accent-red)';

  return createPortal(
    <div style={overlayStyle} onClick={onClose} role="presentation">
      <div
        style={modalStyle}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="workflow-result-title"
      >
        <div style={{
          padding: '16px 18px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          gap: 12,
          alignItems: 'flex-start',
        }}>
          <div>
            <div
              id="workflow-result-title"
              style={{
                color,
                fontSize: 14,
                fontWeight: 900,
                textTransform: 'uppercase',
              }}
            >
              {title}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 5, lineHeight: 1.5 }}>
              {summary}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              ...buttonStyle('var(--text-secondary)'),
              minHeight: 30,
              width: 30,
              minWidth: 30,
              padding: 0,
            }}
            aria-label="Close workflow result"
            title="Close"
          >
            x
          </button>
        </div>

        <div style={{ padding: 18, overflowY: 'auto' }}>
          {outputPath ? (
            <div style={{
              border: `1px solid color-mix(in srgb, ${color} 34%, var(--border-color))`,
              background: `color-mix(in srgb, ${color} 8%, var(--bg-secondary))`,
              borderRadius: 8,
              padding: '10px 12px',
              color,
              fontSize: 11,
              lineHeight: 1.45,
              fontFamily: 'monospace',
              wordBreak: 'break-all',
              marginBottom: 14,
            }}>
              {outputPath}
            </div>
          ) : null}
          <div style={{ display: 'grid', gap: 10 }}>
            <div>
              <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900 }}>
                What to expect
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.5, marginTop: 3 }}>
                {expectation}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900 }}>
                Next step
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.5, marginTop: 3 }}>
                {nextStep}
              </div>
            </div>
          </div>
        </div>

        <div style={{
          padding: '12px 18px 16px',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
        }}>
          <label style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: 'var(--text-secondary)',
            fontSize: 12,
            cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={doNotShowAgain}
              onChange={(event) => onDoNotShowAgainChange(event.target.checked)}
            />
            Do not show again
          </label>
          <button type="button" onClick={onClose} style={buttonStyle(color)}>
            Done
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export default memo(ResultModal);
