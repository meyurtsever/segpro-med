import type { CSSProperties, ReactNode } from 'react';

interface NodeHintProps {
  children: ReactNode;
  style?: CSSProperties;
}

const hintStyle: CSSProperties = {
  marginTop: 8,
  padding: '6px 8px',
  fontSize: 11,
  lineHeight: 1.4,
  color: 'var(--accent-orange)',
  background: 'rgba(245, 181, 79, 0.1)',
  border: '1px solid rgba(245, 181, 79, 0.28)',
  borderRadius: 6,
};

export default function NodeHint({ children, style }: NodeHintProps) {
  return (
    <div style={{ ...hintStyle, ...style }}>
      {children}
    </div>
  );
}
