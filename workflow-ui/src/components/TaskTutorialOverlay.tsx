import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import type { Node } from '@xyflow/react';
import type { BaseNodeData } from '../types/nodes';
import type { TaskTutorialConfig } from '../engine/taskTutorials';

interface HighlightRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

interface TaskTutorialOverlayProps {
  config: TaskTutorialConfig;
  nodes: Node<BaseNodeData>[];
  viewportSignal: number;
  doNotShowAgain: boolean;
  onDoNotShowAgainChange: (checked: boolean) => void;
  onClose: () => void;
}

const padding = 10;

function getNodeElement(nodeId: string): HTMLElement | null {
  return document.querySelector(`.react-flow__node[data-id="${nodeId}"]`) as HTMLElement | null;
}

function TaskTutorialOverlay({
  config,
  nodes,
  viewportSignal,
  doNotShowAgain,
  onDoNotShowAgainChange,
  onClose,
}: TaskTutorialOverlayProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [rect, setRect] = useState<HighlightRect | null>(null);
  const step = config.steps[stepIndex];
  const targetNode = useMemo(() => nodes.find((node) =>
    node.data?.taskTemplateId === config.templateId &&
    node.data?.taskNodeKey === step?.nodeKey,
  ), [config.templateId, nodes, step?.nodeKey]);

  const updateRect = useCallback(() => {
    if (!targetNode || typeof document === 'undefined') {
      setRect(null);
      return;
    }

    const element = getNodeElement(targetNode.id);
    if (!element) {
      setRect(null);
      return;
    }

    const bounds = element.getBoundingClientRect();
    setRect({
      top: bounds.top - padding,
      left: bounds.left - padding,
      width: bounds.width + padding * 2,
      height: bounds.height + padding * 2,
    });
  }, [targetNode]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(updateRect);
    window.addEventListener('resize', updateRect);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', updateRect);
    };
  }, [stepIndex, updateRect, viewportSignal, nodes]);

  if (!step || !rect) return null;

  const total = config.steps.length;
  const isLast = stepIndex === total - 1;
  const cardWidth = 320;
  const preferredPlacement = step.placement || 'right';
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  let cardLeft = rect.left + rect.width + 16;
  let cardTop = rect.top;

  if (preferredPlacement === 'left') {
    cardLeft = rect.left - cardWidth - 16;
  } else if (preferredPlacement === 'bottom') {
    cardLeft = rect.left;
    cardTop = rect.top + rect.height + 16;
  } else if (preferredPlacement === 'top') {
    cardLeft = rect.left;
    cardTop = rect.top - 190;
  }

  if (cardLeft + cardWidth > viewportWidth - 16) {
    cardLeft = Math.max(16, rect.left - cardWidth - 16);
  }
  if (cardLeft < 16) cardLeft = 16;
  if (cardTop + 190 > viewportHeight - 16) {
    cardTop = Math.max(16, viewportHeight - 206);
  }
  if (cardTop < 16) cardTop = 16;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9998,
        pointerEvents: 'auto',
      }}
    >
      <div
        style={{
          position: 'fixed',
          top: rect.top,
          left: rect.left,
          width: rect.width,
          height: rect.height,
          borderRadius: 14,
          border: '2px solid var(--accent-green)',
          boxShadow: '0 0 0 9999px rgba(5, 7, 18, 0.72), 0 0 28px rgba(76, 175, 139, 0.4)',
          pointerEvents: 'none',
          transition: 'top 120ms ease, left 120ms ease, width 120ms ease, height 120ms ease',
        }}
      />
      <div
        style={{
          position: 'fixed',
          top: cardTop,
          left: cardLeft,
          width: cardWidth,
          borderRadius: 10,
          border: '1px solid color-mix(in srgb, var(--accent-green) 46%, var(--border-color))',
          background: 'var(--bg-secondary)',
          boxShadow: '0 18px 52px rgba(0, 0, 0, 0.5)',
          overflow: 'hidden',
        }}
      >
        <div style={{
          padding: '12px 14px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          gap: 10,
          alignItems: 'center',
        }}>
          <div>
            <div style={{ color: 'var(--accent-green)', fontSize: 11, fontWeight: 900 }}>
              {stepIndex + 1}/{total}
            </div>
            <div style={{ color: 'var(--text-primary)', fontSize: 14, fontWeight: 900, marginTop: 3 }}>
              {step.title}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              minWidth: 58,
              height: 30,
              borderRadius: 6,
              border: '1px solid var(--border-color)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: 900,
              padding: '0 10px',
            }}
          >
            Close
          </button>
        </div>
        <div style={{ padding: 14 }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.5 }}>
            {step.body}
          </div>
        </div>
        <div style={{
          padding: '10px 14px 14px',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
        }}>
          <label style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            color: 'var(--text-secondary)',
            fontSize: 11,
            cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={doNotShowAgain}
              onChange={(event) => onDoNotShowAgainChange(event.target.checked)}
            />
            Do not show again
          </label>
          <button
            type="button"
            onClick={() => {
              if (isLast) {
                onClose();
              } else {
                setStepIndex((value) => value + 1);
              }
            }}
            style={{
              padding: '8px 13px',
              borderRadius: 6,
              border: '1px solid color-mix(in srgb, var(--accent-green) 54%, var(--border-color))',
              background: 'color-mix(in srgb, var(--accent-green) 15%, var(--bg-secondary))',
              color: 'var(--accent-green)',
              fontSize: 12,
              fontWeight: 900,
              cursor: 'pointer',
            }}
          >
            {isLast ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default memo(TaskTutorialOverlay);
