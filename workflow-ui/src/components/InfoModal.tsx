/**
 * InfoModal — Shows contextual help about a workflow node.
 * Displays description, expected inputs/outputs, and tips.
 */

import { memo, useEffect } from 'react';
import { createPortal } from 'react-dom';

export interface NodeInfo {
  description: string;
  inputs: string[];
  outputs: string[];
  tips?: string[];
}

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  icon: string;
  color: string;
  info: NodeInfo;
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0, 0, 0, 0.6)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 9999,
  backdropFilter: 'blur(4px)',
};

const modalStyle: React.CSSProperties = {
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-color)',
  borderRadius: 12,
  width: 420,
  maxHeight: '80vh',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
};

const modalBodyStyle: React.CSSProperties = {
  overflowY: 'auto',
  flex: 1,
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  textTransform: 'uppercase',
  letterSpacing: '0.8px',
  marginBottom: 6,
  display: 'flex',
  alignItems: 'center',
  gap: 6,
};

const listItemStyle: React.CSSProperties = {
  fontSize: 12,
  color: 'var(--text-secondary)',
  padding: '3px 0',
  display: 'flex',
  alignItems: 'flex-start',
  gap: 6,
};

function InfoModal({ isOpen, onClose, title, icon, color, info }: InfoModalProps) {
  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div style={overlayStyle} onClick={onClose}>
      <div
        style={modalStyle}
        onClick={(e) => e.stopPropagation()}
        onWheel={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: `1px solid ${color}30`,
            background: `${color}10`,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <span style={{ fontSize: 24 }}>{icon}</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
              {title}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
              Node Documentation
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: 18,
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: 4,
            }}
            onMouseEnter={(e) => {
              (e.target as HTMLElement).style.color = 'var(--text-primary)';
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLElement).style.color = 'var(--text-muted)';
            }}
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={modalBodyStyle}>
        <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Description */}
          <div>
            <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
              {info.description}
            </div>
          </div>

          {/* Inputs */}
          <div>
            <div style={{ ...sectionTitleStyle, color: 'var(--accent-blue)' }}>
              <span>📥</span> Expected Inputs
            </div>
            {info.inputs.map((input, i) => (
              <div key={i} style={listItemStyle}>
                <span style={{ color: 'var(--accent-blue)', fontSize: 8, marginTop: 4 }}>●</span>
                {input}
              </div>
            ))}
          </div>

          {/* Outputs */}
          <div>
            <div style={{ ...sectionTitleStyle, color: 'var(--accent-green)' }}>
              <span>📤</span> Outputs
            </div>
            {info.outputs.map((output, i) => (
              <div key={i} style={listItemStyle}>
                <span style={{ color: 'var(--accent-green)', fontSize: 8, marginTop: 4 }}>●</span>
                {output}
              </div>
            ))}
          </div>

          {/* Tips */}
          {info.tips && info.tips.length > 0 && (
            <div>
              <div style={{ ...sectionTitleStyle, color: 'var(--accent-orange)' }}>
                <span>💡</span> Tips
              </div>
              {info.tips.map((tip, i) => (
                <div key={i} style={listItemStyle}>
                  <span style={{ color: 'var(--accent-orange)', fontSize: 8, marginTop: 4 }}>●</span>
                  {tip}
                </div>
              ))}
            </div>
          )}
        </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export default memo(InfoModal);
