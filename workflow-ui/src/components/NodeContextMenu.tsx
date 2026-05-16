import type { CSSProperties } from 'react';

interface NodeContextMenuAction {
  id: string;
  label: string;
  description: string;
  color: string;
  disabled?: boolean;
  onClick: () => void;
}

interface NodeContextMenuProps {
  title: string;
  icon: string;
  category?: string;
  position: { x: number; y: number };
  actions: NodeContextMenuAction[];
  onClose: () => void;
}

const menuStyle: CSSProperties = {
  position: 'fixed',
  zIndex: 80,
  width: 258,
  pointerEvents: 'all',
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-color)',
  borderRadius: 7,
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const headerStyle: CSSProperties = {
  padding: '9px 10px',
  borderBottom: '1px solid var(--border-color)',
  display: 'flex',
  alignItems: 'center',
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

const itemStyle = (color: string, disabled = false): CSSProperties => ({
  width: '100%',
  border: 'none',
  borderBottom: '1px solid rgba(255,255,255,0.06)',
  background: 'transparent',
  color: disabled ? 'var(--text-muted)' : color,
  padding: '8px 10px',
  cursor: disabled ? 'not-allowed' : 'pointer',
  textAlign: 'left',
  opacity: disabled ? 0.52 : 1,
});

function setHoverBackground(element: HTMLElement, hovered: boolean, disabled = false) {
  if (disabled) return;
  element.style.background = hovered ? 'var(--bg-tertiary)' : 'transparent';
}

export default function NodeContextMenu({
  title,
  icon,
  category,
  position,
  actions,
  onClose,
}: NodeContextMenuProps) {
  return (
    <div
      style={{
        ...menuStyle,
        left: position.x,
        top: position.y,
      }}
      className="nodrag nopan nowheel"
      onPointerDown={(event) => event.stopPropagation()}
      onMouseDown={(event) => event.stopPropagation()}
      onClick={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.preventDefault()}
      onWheel={(event) => event.stopPropagation()}
      role="menu"
    >
      <div style={headerStyle}>
        <span style={{ fontSize: 16, lineHeight: 1 }}>{icon}</span>
        <span style={{ minWidth: 0, flex: 1 }}>
          <span
            style={{
              display: 'block',
              color: 'var(--text-primary)',
              fontSize: 12,
              fontWeight: 800,
              lineHeight: 1.25,
            }}
          >
            {title}
          </span>
          {category ? (
            <span
              style={{
                display: 'block',
                color: 'var(--text-muted)',
                fontSize: 9,
                fontWeight: 800,
                letterSpacing: 0.5,
                textTransform: 'uppercase',
                marginTop: 2,
              }}
            >
              {category}
            </span>
          ) : null}
        </span>
        <button
          type="button"
          onClick={onClose}
          style={closeButtonStyle}
          title="Close node actions"
          aria-label="Close node actions"
        >
          x
        </button>
      </div>

      {actions.map((action, index) => (
        <button
          key={action.id}
          type="button"
          disabled={action.disabled}
          onClick={() => {
            if (action.disabled) return;
            action.onClick();
          }}
          style={{
            ...itemStyle(action.color, action.disabled),
            borderBottom: index === actions.length - 1 ? 'none' : itemStyle(action.color).borderBottom,
          }}
          onMouseEnter={(event) => setHoverBackground(event.currentTarget, true, action.disabled)}
          onMouseLeave={(event) => setHoverBackground(event.currentTarget, false, action.disabled)}
          role="menuitem"
        >
          <span style={{ display: 'block', fontSize: 12, fontWeight: 800 }}>
            {action.label}
          </span>
          <span
            style={{
              display: 'block',
              color: 'var(--text-muted)',
              fontSize: 10,
              lineHeight: 1.35,
              marginTop: 2,
            }}
          >
            {action.description}
          </span>
        </button>
      ))}
    </div>
  );
}
