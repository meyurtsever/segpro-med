/**
 * InteractiveAnnotatorNode
 * =========================
 * Full-featured canvas-based annotation node for medical imaging workflows.
 *
 * Drawing tools: Polygon, Circle, Freehand, Point (coordinate selection)
 * Navigation  : Zoom (wheel + buttons), Pan (drag in pan mode), Slice nav
 * Fullscreen  : Expands to a portal overlay while preserving annotation positions
 *
 * Architecture (adapted from SegPro-Med's custom Gradio annotator plugin):
 *   - Dual coordinate system: image-space (persisted) ↔ display-space (rendered)
 *   - Canvas→Image: imageX = (canvasX - offsetX) / scale
 *   - Image→Canvas: canvasX = imageX * scale + offsetX
 *   - All annotations store image-space coordinates and are re-projected on every render
 *   - Zoom/pan/fullscreen only change scale + offsets → annotations never move
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type {
  InteractiveAnnotatorNodeData,
  AutoSegmentationNodeData,
  AnnotationShape,
  AnnotationTool,
  ImagePoint,
  SegmentationPrompt,
} from '../types/nodes';
import * as api from '../api/client';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 12;
const ZOOM_FACTOR = 1.1;
const POINT_MARKER_RADIUS = 6;
const MIN_FREEHAND_DIST = 2; // px in image-space
const ANNOTATION_ALPHA = 0.3;
const VIEWS = ['axial', 'coronal', 'sagittal'] as const;

const ANNOTATION_COLORS = [
  '#4fc3f7', '#81c784', '#ff8a65', '#ba68c8',
  '#ffd54f', '#e57373', '#4db6ac', '#7986cb',
];

// ---------------------------------------------------------------------------
// Info modal content
// ---------------------------------------------------------------------------

const ANNOTATOR_INFO: NodeInfo = {
  description:
    'Interactive annotation tool for medical imaging slices. Draw polygons, circles, freehand paths, and place coordinate markers directly on slices. Annotations persist across zoom and fullscreen transitions using image-space coordinates.',
  inputs: [
    'Session ID from a Data Loader node (direct connection)',
    'Output file path from a Format Converter node (auto-loads on execution)',
  ],
  outputs: [
    'Session ID — passes through to downstream nodes',
    'Annotations array — all shapes in image-space coordinates',
  ],
  tips: [
    '🔷 Polygon: Click to place vertices, press Space or click start point to close',
    '⭕ Circle: Click and drag from center outward',
    '✏️ Freehand: Click and drag to draw freeform paths',
    '📍 Point: Click to place coordinate markers',
    '🖐️ Pan: Drag to move the view (or use any tool + middle mouse)',
    '🔍 Zoom: Scroll wheel on the canvas, or use −/+ buttons',
    '⛶ Fullscreen: Press the expand button — annotations keep their positions',
    'Rectangle: Drag to draw a rectangle annotation',
  ],
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const viewerContainerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
};

const toolbarStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 3,
  alignItems: 'center',
};

const toolBtnBase: React.CSSProperties = {
  padding: '4px 5px',
  borderRadius: 4,
  border: '1px solid var(--border-color)',
  fontSize: 9,
  fontWeight: 600,
  cursor: 'pointer',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  transition: 'all 0.15s ease',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 2,
  lineHeight: 1,
  width: 46,
  minWidth: 46,
  overflow: 'hidden',
};

const activeTool: React.CSSProperties = {
  ...toolBtnBase,
  background: 'var(--accent-green)',
  color: '#fff',
  borderColor: 'var(--accent-green)',
};

const canvasWrapperStyle: React.CSSProperties = {
  position: 'relative',
  width: '100%',
  aspectRatio: '1',
  background: '#000',
  borderRadius: 6,
  overflow: 'hidden',
  border: '1px solid var(--border-color)',
};

const canvasStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
};

const controlsRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
};

const navBtnStyle: React.CSSProperties = {
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-color)',
  borderRadius: 4,
  color: 'var(--text-primary)',
  cursor: 'pointer',
  padding: '4px 8px',
  fontSize: 12,
  fontWeight: 600,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  minWidth: 28,
  transition: 'all 0.15s ease',
};

const sliderStyle: React.CSSProperties = {
  flex: 1,
  height: 4,
  accentColor: 'var(--accent-green)',
  cursor: 'pointer',
};

const sliceInfoStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  fontFamily: 'monospace',
  textAlign: 'center',
  minWidth: 65,
  flexShrink: 0,
};

const viewSelectorStyle: React.CSSProperties = {
  display: 'flex',
  gap: 4,
};

const viewBtnBase: React.CSSProperties = {
  flex: 1,
  padding: '4px 6px',
  borderRadius: 4,
  border: '1px solid var(--border-color)',
  fontSize: 10,
  fontWeight: 600,
  cursor: 'pointer',
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  transition: 'all 0.15s ease',
  textAlign: 'center',
};

const metaRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  fontSize: 10,
  color: 'var(--text-muted)',
  fontFamily: 'monospace',
};

const smallBtnStyle: React.CSSProperties = {
  ...navBtnStyle,
  padding: '2px 6px',
  fontSize: 11,
  minWidth: 24,
};

const badgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '2px 6px',
  borderRadius: 4,
  fontSize: 10,
  fontFamily: 'monospace',
  background: 'rgba(76, 175, 80, 0.08)',
  border: '1px solid rgba(76, 175, 80, 0.25)',
  color: 'var(--accent-green)',
};

const contextMenuItemStyle: React.CSSProperties = {
  display: 'block',
  width: '100%',
  padding: '7px 14px',
  background: 'transparent',
  border: 'none',
  color: 'var(--text-primary)',
  fontSize: 12,
  cursor: 'pointer',
  textAlign: 'left',
  borderRadius: 0,
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 2,
  display: 'block',
};

// ---------------------------------------------------------------------------
// Fullscreen overlay styles
// ---------------------------------------------------------------------------

const fullscreenOverlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 9999,
  background: '#0d0e1a',
  display: 'flex',
  flexDirection: 'column',
};

const fullscreenHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '8px 16px',
  background: 'rgba(26, 27, 46, 0.95)',
  borderBottom: '1px solid var(--border-color)',
  gap: 12,
  flexShrink: 0,
};

const fullscreenCanvasArea: React.CSSProperties = {
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
};

// ---------------------------------------------------------------------------
// Helper: convert hex color to rgba string
// ---------------------------------------------------------------------------
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ---------------------------------------------------------------------------
// SVG Icons (ported from Gradio image annotator plugin)
// ---------------------------------------------------------------------------
const IconRect = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="5" width="18" height="14" rx="1" />
  </svg>
);
const IconPolygon = (
  <svg width="16" height="16" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="7,7 15,9 21,5 25,13 23,21 15,19 9,23 5,15" />
    <circle cx="7" cy="7" r="2" fill="currentColor" /><circle cx="15" cy="9" r="2" fill="currentColor" />
    <circle cx="21" cy="5" r="2" fill="currentColor" /><circle cx="25" cy="13" r="2" fill="currentColor" />
    <circle cx="23" cy="21" r="2" fill="currentColor" /><circle cx="15" cy="19" r="2" fill="currentColor" />
    <circle cx="9" cy="23" r="2" fill="currentColor" /><circle cx="5" cy="15" r="2" fill="currentColor" />
  </svg>
);
const IconCircle = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="12" cy="12" r="10" />
  </svg>
);
const IconFreehand = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 17c1-3 4-6 8-6s5 3 8 3 2-2 2-4" />
    <path d="M3 21c1-2 4-4 8-4s5 2 8 2" />
  </svg>
);
const IconPoint = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <circle cx="12" cy="12" r="3" />
    <line x1="12" y1="2" x2="12" y2="8" /><line x1="12" y1="16" x2="12" y2="22" />
    <line x1="2" y1="12" x2="8" y2="12" /><line x1="16" y1="12" x2="22" y2="12" />
  </svg>
);
const IconPan = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2" />
    <path d="M14 10V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v2" />
    <path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8" />
    <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
  </svg>
);
const IconClear = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 6h18" /><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    <line x1="9" y1="10" x2="15" y2="16" /><line x1="15" y1="10" x2="9" y2="16" />
  </svg>
);
const IconDownload = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);
const IconUndo = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 14 4 9 9 4" />
    <path d="M20 20v-7a4 4 0 0 0-4-4H4" />
  </svg>
);
const IconRedo = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 14 20 9 15 4" />
    <path d="M4 20v-7a4 4 0 0 1 4-4h12" />
  </svg>
);
const IconEraser = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 20H7L3 16l10-10 7 7-3.5 3.5" />
    <path d="M6.5 17.5l4-4" />
  </svg>
);
const IconNavigator = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="2" width="20" height="20" rx="2" />
    <rect x="6" y="6" width="8" height="8" rx="1" fill="currentColor" opacity="0.3" />
    <line x1="10" y1="3" x2="10" y2="6" /><line x1="10" y1="14" x2="10" y2="21" />
    <line x1="3" y1="10" x2="6" y2="10" /><line x1="14" y1="10" x2="21" y2="10" />
  </svg>
);
const IconLabels = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 7V4h16v3" /><line x1="12" y1="4" x2="12" y2="20" />
    <path d="M8 20h8" />
  </svg>
);

// ---------------------------------------------------------------------------
// Predefined label list (matching SegPro-Med crowdsourcing labels)
// ---------------------------------------------------------------------------
const PREDEFINED_LABELS = ['Tumor', 'Normal Tissue', 'Organ', 'Lesion', 'ROI', 'Other'];
const LABEL_COLOR_MAP: Record<string, string> = {
  'Tumor': '#e57373',
  'Normal Tissue': '#81c784',
  'Organ': '#4fc3f7',
  'Lesion': '#ffd54f',
  'ROI': '#ba68c8',
  'Other': '#4db6ac',
};

// Popup color palette for custom selection
const COLOR_PALETTE = [
  '#e57373', '#81c784', '#4fc3f7', '#ffd54f', '#ba68c8',
  '#4db6ac', '#ff8a65', '#7986cb', '#fff176', '#f48fb1',
  '#80cbc4', '#ce93d8', '#a1887f', '#90a4ae', '#ffab91',
];

// ---------------------------------------------------------------------------
// LabelEditorPopup – shown after shape draw or on double-click to edit
// ---------------------------------------------------------------------------
function LabelEditorPopup({
  initialLabel,
  initialColor,
  onConfirm,
  onCancel,
}: {
  initialLabel: string;
  initialColor: string;
  onConfirm: (label: string, color: string) => void;
  onCancel: () => void;
}) {
  const [label, setLabel] = useState(initialLabel);
  const [color, setColor] = useState(initialColor);
  const [customText, setCustomText] = useState(
    PREDEFINED_LABELS.includes(initialLabel) ? '' : initialLabel,
  );
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Auto-focus custom text or the first element
    const timer = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(timer);
  }, []);

  const handleSelectPredefined = (l: string) => {
    setLabel(l);
    setCustomText('');
    if (LABEL_COLOR_MAP[l]) setColor(LABEL_COLOR_MAP[l]);
  };

  const handleCustomTextChange = (val: string) => {
    setCustomText(val);
    if (val.trim()) setLabel(val.trim());
  };

  const handleConfirm = () => {
    const finalLabel = customText.trim() || label || 'Unlabeled';
    onConfirm(finalLabel, color);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'Enter') handleConfirm();
    if (e.key === 'Escape') onCancel();
  };

  return (
    <div
      style={{
        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(0,0,0,0.5)', zIndex: 50,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
      onKeyDown={handleKeyDown}
    >
      <div
        style={{
          background: 'var(--bg-secondary, #1a1b2e)', border: '1px solid var(--border-color, #333)',
          borderRadius: 8, padding: 16, minWidth: 260, maxWidth: 320,
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Title */}
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary, #fff)', marginBottom: 12 }}>
          Annotation Label
        </div>

        {/* Predefined labels as pill buttons */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
          {PREDEFINED_LABELS.map((l) => (
            <button
              key={l}
              onClick={() => handleSelectPredefined(l)}
              style={{
                padding: '3px 8px', borderRadius: 12, fontSize: 10, fontWeight: 600,
                cursor: 'pointer', transition: 'all 0.15s',
                border: label === l && !customText ? '2px solid var(--accent-green, #4caf50)' : '1px solid var(--border-color, #444)',
                background: label === l && !customText ? 'var(--accent-green, #4caf50)' : 'var(--bg-tertiary, #252640)',
                color: label === l && !customText ? '#fff' : 'var(--text-secondary, #aaa)',
              }}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Custom text input */}
        <input
          ref={inputRef}
          type="text"
          value={customText}
          onChange={(e) => handleCustomTextChange(e.target.value)}
          placeholder="Or type custom label…"
          onKeyDown={handleKeyDown}
          style={{
            width: '100%', padding: '6px 8px', borderRadius: 4, fontSize: 11,
            border: '1px solid var(--border-color, #444)',
            background: 'var(--bg-primary, #0d0e1a)', color: 'var(--text-primary, #fff)',
            outline: 'none', boxSizing: 'border-box',
          }}
        />

        {/* Color picker */}
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted, #888)', marginBottom: 4 }}>Color</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {COLOR_PALETTE.map((c) => (
              <button
                key={c}
                onClick={() => setColor(c)}
                style={{
                  width: 20, height: 20, borderRadius: '50%', cursor: 'pointer',
                  background: c, border: c === color ? '2px solid #fff' : '2px solid transparent',
                  boxShadow: c === color ? '0 0 4px rgba(255,255,255,0.5)' : 'none',
                  padding: 0, transition: 'all 0.1s',
                }}
                title={c}
              />
            ))}
          </div>
        </div>

        {/* Preview & actions */}
        <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            flex: 1, padding: '4px 8px', borderRadius: 4, fontSize: 11,
            background: hexToRgba(color, 0.15), border: `1px solid ${color}`, color,
            fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {customText.trim() || label || 'Unlabeled'}
          </div>
          <button
            onClick={onCancel}
            style={{
              padding: '5px 10px', borderRadius: 4, fontSize: 11, fontWeight: 600,
              cursor: 'pointer', border: '1px solid var(--border-color, #444)',
              background: 'var(--bg-tertiary, #252640)', color: 'var(--text-secondary, #aaa)',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            style={{
              padding: '5px 12px', borderRadius: 4, fontSize: 11, fontWeight: 700,
              cursor: 'pointer', border: 'none',
              background: 'var(--accent-green, #4caf50)', color: '#fff',
            }}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ViewState: zoom + pan (equivalent to WindowViewer in Gradio plugin)
// ---------------------------------------------------------------------------
interface ViewState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

// ---------------------------------------------------------------------------
// Position Navigator – mini-map preview (ported from PositionNavigator.svelte)
// ---------------------------------------------------------------------------
const NAV_MAX_SIZE = 180;
const NAV_MIN_SIZE = 80;

function PositionNavigator({
  imageObj,
  imgDimensions,
  viewState,
  canvasRef,
  onNavigate,
}: {
  imageObj: HTMLImageElement | null;
  imgDimensions: { w: number; h: number };
  viewState: ViewState;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  onNavigate: (offsetX: number, offsetY: number) => void;
}) {
  const navCanvasRef = useRef<HTMLCanvasElement>(null);
  const navDragRef = useRef(false);

  const navDims = useMemo(() => {
    const { w, h } = imgDimensions;
    const aspect = w / h;
    let nw: number, nh: number;
    if (aspect >= 1) {
      nw = NAV_MAX_SIZE;
      nh = nw / aspect;
    } else {
      nh = NAV_MAX_SIZE;
      nw = nh * aspect;
    }
    return { width: Math.max(NAV_MIN_SIZE, Math.round(nw)), height: Math.max(NAV_MIN_SIZE, Math.round(nh)) };
  }, [imgDimensions]);

  // Redraw navigator canvas on every viewState / image change
  useEffect(() => {
    const canvas = navCanvasRef.current;
    if (!canvas) return;
    canvas.width = navDims.width;
    canvas.height = navDims.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const navScale = Math.min(navDims.width / imgDimensions.w, navDims.height / imgDimensions.h);
    ctx.clearRect(0, 0, navDims.width, navDims.height);

    // Thumbnail
    if (imageObj) {
      ctx.drawImage(imageObj, 0, 0, imgDimensions.w * navScale, imgDimensions.h * navScale);
    } else {
      ctx.fillStyle = '#333';
      ctx.fillRect(0, 0, navDims.width, navDims.height);
    }

    // Viewport rectangle
    const cw = canvasRef.current?.width || 1;
    const ch = canvasRef.current?.height || 1;
    const visX = -viewState.offsetX / viewState.scale;
    const visY = -viewState.offsetY / viewState.scale;
    const visW = cw / viewState.scale;
    const visH = ch / viewState.scale;
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
    ctx.fillRect(visX * navScale, visY * navScale, visW * navScale, visH * navScale);
    ctx.strokeRect(visX * navScale, visY * navScale, visW * navScale, visH * navScale);
  }, [imageObj, imgDimensions, viewState, navDims, canvasRef]);

  const handleNavClick = useCallback(
    (e: React.MouseEvent) => {
      const canvas = navCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const navScale = Math.min(navDims.width / imgDimensions.w, navDims.height / imgDimensions.h);
      const imgX = (e.clientX - rect.left) / navScale;
      const imgY = (e.clientY - rect.top) / navScale;
      const cw = canvasRef.current?.width || 1;
      const ch = canvasRef.current?.height || 1;
      onNavigate(cw / 2 - imgX * viewState.scale, ch / 2 - imgY * viewState.scale);
    },
    [navDims, imgDimensions, viewState.scale, canvasRef, onNavigate],
  );

  const handleNavPointerDown = useCallback(
    (e: React.PointerEvent) => { e.stopPropagation(); navDragRef.current = true; handleNavClick(e); },
    [handleNavClick],
  );
  const handleNavPointerMove = useCallback(
    (e: React.PointerEvent) => { e.stopPropagation(); if (navDragRef.current) handleNavClick(e); },
    [handleNavClick],
  );
  const handleNavPointerUp = useCallback(
    (e: React.PointerEvent) => { e.stopPropagation(); navDragRef.current = false; },
    [],
  );

  return (
    <div
      style={{
        position: 'absolute', bottom: 8, right: 8,
        border: '2px solid rgba(59, 130, 246, 0.6)', borderRadius: 4,
        overflow: 'hidden', background: 'rgba(0, 0, 0, 0.75)',
        cursor: 'crosshair', zIndex: 10,
        boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
      }}
      onPointerDown={handleNavPointerDown}
      onPointerMove={handleNavPointerMove}
      onPointerUp={handleNavPointerUp}
      onPointerLeave={handleNavPointerUp}
    >
      <canvas ref={navCanvasRef} style={{ display: 'block' }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Returns 0–2 intersection points of segment p1→p2 with a circle. */
function lineCircleIntersect(
  p1: ImagePoint, p2: ImagePoint,
  c: ImagePoint, r: number,
): ImagePoint[] {
  const dx = p2.x - p1.x, dy = p2.y - p1.y;
  const fx = p1.x - c.x, fy = p1.y - c.y;
  const a = dx * dx + dy * dy;
  const b = 2 * (fx * dx + fy * dy);
  const cc = fx * fx + fy * fy - r * r;
  const disc = b * b - 4 * a * cc;
  if (disc < 0 || a === 0) return [];
  const sqrtDisc = Math.sqrt(disc);
  const res: ImagePoint[] = [];
  for (const sign of [-1, 1]) {
    const t = (-b + sign * sqrtDisc) / (2 * a);
    if (t >= 0 && t <= 1) res.push({ x: p1.x + t * dx, y: p1.y + t * dy });
  }
  return res;
}

/** Returns the polygon that remains after clipping away the circle region,
 *  or null if the entire polygon is consumed. */
function clipPolygonByCircle(pts: ImagePoint[], center: ImagePoint, r: number): ImagePoint[] | null {
  if (pts.length < 3) return null;
  const result: ImagePoint[] = [];
  const n = pts.length;
  const r2 = r * r;
  for (let i = 0; i < n; i++) {
    const cur = pts[i];
    const nxt = pts[(i + 1) % n];
    const curIn = (cur.x - center.x) ** 2 + (cur.y - center.y) ** 2 <= r2;
    const nxtIn = (nxt.x - center.x) ** 2 + (nxt.y - center.y) ** 2 <= r2;
    if (!curIn) result.push(cur);
    if (curIn !== nxtIn) {
      // Sort intersection points along the edge direction
      const pts2 = lineCircleIntersect(cur, nxt, center, r);
      if (!curIn) {
        // Going inside: push the entry point (first t along cur→nxt)
        pts2.sort((a, b) => (a.x - cur.x) ** 2 + (a.y - cur.y) ** 2 - ((b.x - cur.x) ** 2 + (b.y - cur.y) ** 2));
        if (pts2.length) result.push(pts2[0]);
      } else {
        // Coming outside: push the exit point (last t along cur→nxt)
        pts2.sort((a, b) => (a.x - cur.x) ** 2 + (a.y - cur.y) ** 2 - ((b.x - cur.x) ** 2 + (b.y - cur.y) ** 2));
        if (pts2.length) result.push(pts2[pts2.length - 1]);
      }
    }
  }
  return result.length >= 3 ? result : null;
}

/**
 * Douglas-Peucker polygon simplification.
 * Removes points whose perpendicular distance from the chord through their neighbours
 * is less than `epsilon` (in image-space pixels).
 * Prevents runaway point accumulation when the eraser is dragged across a shape.
 */
function simplifyPolygon(pts: ImagePoint[], epsilon: number): ImagePoint[] {
  if (pts.length <= 2) return pts;
  const p1 = pts[0], pN = pts[pts.length - 1];
  const dx = pN.x - p1.x, dy = pN.y - p1.y;
  const lenSq = dx * dx + dy * dy;
  let maxDist = 0, maxIdx = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const dist = lenSq > 0
      ? Math.abs(dy * pts[i].x - dx * pts[i].y + pN.x * p1.y - pN.y * p1.x) / Math.sqrt(lenSq)
      : Math.sqrt((pts[i].x - p1.x) ** 2 + (pts[i].y - p1.y) ** 2);
    if (dist > maxDist) { maxDist = dist; maxIdx = i; }
  }
  if (maxDist > epsilon) {
    const left = simplifyPolygon(pts.slice(0, maxIdx + 1), epsilon);
    const right = simplifyPolygon(pts.slice(maxIdx), epsilon);
    return [...left.slice(0, -1), ...right];
  }
  return [p1, pN];
}

/** Applies eraser geometry to a shape.
 *  - polygon / freehand: polygon-clip (removes inside portion)
 *  - rect              : convert to polygon then clip
 *  - circle            : approximate as 32-gon then clip
 *  - point             : delete if inside eraser circle
 *  Returns the (possibly modified) shape, or null if the shape should be removed. */
function eraseShapeByCircle(
  ann: AnnotationShape,
  center: ImagePoint,
  r: number,
): AnnotationShape | null {
  // Simplification tolerance in image-space pixels — removes collinear duplicates
  // that accumulate when the eraser is dragged across a shape over many move events.
  const SIMPLIFY_EPS = 0.5;
  switch (ann.type) {
    case 'polygon':
    case 'freehand': {
      const clipped = clipPolygonByCircle(ann.points, center, r);
      if (!clipped) return null;
      const pts = simplifyPolygon(clipped, SIMPLIFY_EPS);
      return pts.length >= 3 ? { ...ann, points: pts } : null;
    }
    case 'rect': {
      const rectPts: ImagePoint[] = [
        { x: ann.x,              y: ann.y },
        { x: ann.x + ann.width,  y: ann.y },
        { x: ann.x + ann.width,  y: ann.y + ann.height },
        { x: ann.x,              y: ann.y + ann.height },
      ];
      const clipped = clipPolygonByCircle(rectPts, center, r);
      if (!clipped) return null;
      const pts = simplifyPolygon(clipped, SIMPLIFY_EPS);
      if (pts.length < 3) return null;
      // Downgrade rect to polygon since it may no longer be axis-aligned
      return { type: 'polygon', points: pts, label: ann.label, color: ann.color };
    }
    case 'circle': {
      const N = 32;
      const circlePts: ImagePoint[] = Array.from({ length: N }, (_, i) => ({
        x: ann.centerX + ann.radius * Math.cos((2 * Math.PI * i) / N),
        y: ann.centerY + ann.radius * Math.sin((2 * Math.PI * i) / N),
      }));
      const clipped = clipPolygonByCircle(circlePts, center, r);
      if (!clipped) return null;
      const pts = simplifyPolygon(clipped, SIMPLIFY_EPS);
      if (pts.length < 3) return null;
      return { type: 'polygon', points: pts, label: ann.label, color: ann.color };
    }
    case 'point': {
      const dx = ann.x - center.x, dy = ann.y - center.y;
      return dx * dx + dy * dy <= r * r ? null : ann;
    }
    default:
      return ann;
  }
}

function InteractiveAnnotatorNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const storeNodes = useWorkflowStore((s) => s.nodes);
  const storeEdges = useWorkflowStore((s) => s.edges);
  const d = data as unknown as InteractiveAnnotatorNodeData;

  // --- Detect connected Segmentation Profile node ---
  const connectedAutoSeg = useMemo(() => {
    // Find edges where this node is the target and source is autoSegmentation
    for (const edge of storeEdges) {
      if (edge.target === id) {
        const srcNode = storeNodes.find((n) => n.id === edge.source);
        if (srcNode && srcNode.type === 'autoSegmentation') {
          return srcNode.data as unknown as AutoSegmentationNodeData;
        }
      }
    }
    return null;
  }, [id, storeNodes, storeEdges]);

  // --- State ---
  const [loading, setLoading] = useState(false);
  const [segRunning, setSegRunning] = useState(false);
  const [segMessage, setSegMessage] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const [viewState, setViewState] = useState<ViewState>({ scale: 1, offsetX: 0, offsetY: 0 });

  // Drawing state (refs to avoid re-renders during drawing)
  const drawingRef = useRef<{
    active: boolean;
    tool: AnnotationTool;
    points: ImagePoint[];       // polygon / freehand vertices (image-space)
    startPoint: ImagePoint | null; // circle center
    currentRadius: number;
    currentMouse: ImagePoint | null; // live mouse position (image-space)
  }>({
    active: false,
    tool: 'rect',
    points: [],
    startPoint: null,
    currentRadius: 0,
    currentMouse: null,
  });

  // Pan state
  const panRef = useRef<{ active: boolean; startX: number; startY: number; startOX: number; startOY: number }>({
    active: false,
    startX: 0,
    startY: 0,
    startOX: 0,
    startOY: 0,
  });

  // Drag state (for moving annotations in pan mode)
  const dragRef = useRef<{
    active: boolean;
    annIdx: number;
    startImg: ImagePoint;
    originalAnn: AnnotationShape | null;
  }>({ active: false, annIdx: -1, startImg: { x: 0, y: 0 }, originalAnn: null });

  // Vertex drag state (for reshaping single vertices of selected shape)
  const vertexDragRef = useRef<{
    active: boolean;
    annIdx: number;
    vertexIdx: number;       // index within polygon/freehand points, or corner index for rect/circle
    originalAnn: AnnotationShape | null;
  }>({ active: false, annIdx: -1, vertexIdx: -1, originalAnn: null });

  // Undo / Redo stacks keyed by sliceIndex so each slice has its own history
  const undoStackRef = useRef<Record<number, AnnotationShape[][]>>({});
  const redoStackRef = useRef<Record<number, AnnotationShape[][]>>({});

  /** Call BEFORE any mutation that should be undo-able */
  const pushHistory = useCallback((current: AnnotationShape[]) => {
    const si = d.sliceIndex;
    if (!undoStackRef.current[si]) undoStackRef.current[si] = [];
    undoStackRef.current[si].push(JSON.parse(JSON.stringify(current)));
    if (undoStackRef.current[si].length > 50) undoStackRef.current[si].shift();
    if (!redoStackRef.current[si]) redoStackRef.current[si] = [];
    redoStackRef.current[si] = []; // new action clears redo for this slice
  }, [d.sliceIndex]);

  // Eraser size (in image pixels) — stored in local state, not persisted
  const [eraserRadius, setEraserRadius] = useState(4);
  const [showEraserPicker, setShowEraserPicker] = useState(false);
  // Track eraser cursor position for live circle preview
  const eraserCursorRef = useRef<{ x: number; y: number } | null>(null);

  // Right-click context menu for annotations
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; annIdx: number } | null>(null);

  // Position Navigator toggle
  const [showNavigator, setShowNavigator] = useState(false);

  // Label editor popup state
  const [labelEditor, setLabelEditor] = useState<{
    open: boolean;
    mode: 'new' | 'edit';
    pendingShape: AnnotationShape | null;
    editIdx: number;  // index in annotations array when editing
  }>({ open: false, mode: 'new', pendingShape: null, editIdx: -1 });

  const showLabels = d.showLabels !== false; // default true

  const [imgDimensions, setImgDimensions] = useState<{ w: number; h: number } | null>(null);
  const imageObjRef = useRef<HTMLImageElement | null>(null);

  // Canvas refs
  const canvasNodeRef = useRef<HTMLCanvasElement>(null);
  const canvasFSRef = useRef<HTMLCanvasElement>(null);
  const fetchRef = useRef(0);
  const animFrameRef = useRef(0);

  const currentTool = d.activeTool || 'rect';
  const currentView = d.view || 'axial';
  const hasSession = Boolean(d.sessionId);
  const annotations = d.annotations || [];

  const buildPromptFromAnnotation = useCallback(
    (annotation: AnnotationShape | undefined): SegmentationPrompt | null => {
      if (!annotation || !d.sessionId) return null;

      if (annotation.type === 'point') {
        return {
          sessionId: d.sessionId,
          sliceIndex: d.sliceIndex,
          view: currentView,
          points: [{ x: Math.round(annotation.x), y: Math.round(annotation.y), label: 1 }],
          boxes: [],
          source: 'interactiveAnnotator',
        };
      }

      if (annotation.type === 'rect') {
        return {
          sessionId: d.sessionId,
          sliceIndex: d.sliceIndex,
          view: currentView,
          points: [],
          boxes: [{
            x1: Math.round(annotation.x),
            y1: Math.round(annotation.y),
            x2: Math.round(annotation.x + annotation.width),
            y2: Math.round(annotation.y + annotation.height),
          }],
          source: 'interactiveAnnotator',
        };
      }

      return null;
    },
    [currentView, d.sessionId, d.sliceIndex],
  );

  const activePrompt = useMemo(() => {
    const selectedPrompt = buildPromptFromAnnotation(annotations[selectedIdx]);
    if (selectedPrompt) return selectedPrompt;

    for (let index = annotations.length - 1; index >= 0; index -= 1) {
      const prompt = buildPromptFromAnnotation(annotations[index]);
      if (prompt) return prompt;
    }

    return null;
  }, [annotations, buildPromptFromAnnotation, selectedIdx]);

  // --- Load image object from base64 ---
  useEffect(() => {
    if (!d.imageBase64) {
      imageObjRef.current = null;
      return;
    }
    const img = new Image();
    img.src = `data:image/png;base64,${d.imageBase64}`;
    img.onload = () => {
      imageObjRef.current = img;
      setImgDimensions({ w: img.naturalWidth, h: img.naturalHeight });
    };
  }, [d.imageBase64]);

  // --- Coordinate conversion (WindowViewer pattern) ---
  const imageToCanvas = useCallback(
    (ix: number, iy: number, vs: ViewState): { cx: number; cy: number } => ({
      cx: ix * vs.scale + vs.offsetX,
      cy: iy * vs.scale + vs.offsetY,
    }),
    [],
  );

  const canvasToImage = useCallback(
    (cx: number, cy: number, vs: ViewState): ImagePoint => ({
      x: (cx - vs.offsetX) / vs.scale,
      y: (cy - vs.offsetY) / vs.scale,
    }),
    [],
  );

  // --- Fit image to canvas (called on mount, image load, resize, fullscreen toggle) ---
  const fitImageToCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas || !imgDimensions) return;
      const cw = canvas.width;
      const ch = canvas.height;
      const iw = imgDimensions.w;
      const ih = imgDimensions.h;
      const scale = Math.min(cw / iw, ch / ih);
      const offsetX = (cw - iw * scale) / 2;
      const offsetY = (ch - ih * scale) / 2;
      setViewState({ scale, offsetX, offsetY });
    },
    [imgDimensions],
  );

  // --- Canvas size sync ---
  const syncCanvasSize = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas) return;
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const dpr = 1; // use 1:1 for simplicity, consistent with Gradio plugin
      const w = Math.round(rect.width * dpr);
      const h = Math.round(rect.height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    },
    [],
  );

  // --- Rendering ---
  const renderCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const vs = viewState;
      const img = imageObjRef.current;

      // Clear
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw image
      if (img && imgDimensions) {
        ctx.save();
        ctx.translate(vs.offsetX, vs.offsetY);
        ctx.scale(vs.scale, vs.scale);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, imgDimensions.w, imgDimensions.h);
        ctx.restore();
      }

      // Draw all annotations
      for (let i = 0; i < annotations.length; i++) {
        const ann = annotations[i];
        const isSelected = i === selectedIdx;
        drawAnnotation(ctx, ann, vs, isSelected, showLabels);
      }

      // Draw in-progress shape
      const dr = drawingRef.current;
      if (dr.active) {
        drawLiveShape(ctx, dr, vs);
      }

      // Eraser cursor preview
      if (currentTool === 'eraser' && eraserCursorRef.current) {
        const ec = imageToCanvas(eraserCursorRef.current.x, eraserCursorRef.current.y, vs);
        const rCanvas = eraserRadius * vs.scale;
        ctx.save();
        ctx.beginPath();
        ctx.arc(ec.cx, ec.cy, rCanvas, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 80, 80, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.fillStyle = 'rgba(255, 80, 80, 0.08)';
        ctx.fill();
        ctx.restore();
      }
    },
    [viewState, annotations, selectedIdx, imgDimensions, showLabels, currentTool, eraserRadius, imageToCanvas],
  );

  // --- Draw a single annotation ---
  const drawAnnotation = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      ann: AnnotationShape,
      vs: ViewState,
      isSelected: boolean,
      labelsVisible: boolean,
    ) => {
      const lineWidth = isSelected ? 3 : 2;

      switch (ann.type) {
        case 'polygon': {
          if (ann.points.length < 2) break;
          ctx.beginPath();
          const p0 = imageToCanvas(ann.points[0].x, ann.points[0].y, vs);
          ctx.moveTo(p0.cx, p0.cy);
          for (let i = 1; i < ann.points.length; i++) {
            const p = imageToCanvas(ann.points[i].x, ann.points[i].y, vs);
            ctx.lineTo(p.cx, p.cy);
          }
          ctx.closePath();
          ctx.fillStyle = hexToRgba(ann.color, ANNOTATION_ALPHA);
          ctx.fill();
          ctx.strokeStyle = ann.color;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
          // Vertex dots
          if (isSelected) {
            for (const pt of ann.points) {
              const c = imageToCanvas(pt.x, pt.y, vs);
              ctx.fillStyle = ann.color;
              ctx.fillRect(c.cx - 3, c.cy - 3, 6, 6);
            }
          }
          break;
        }
        case 'circle': {
          const center = imageToCanvas(ann.centerX, ann.centerY, vs);
          const rCanvas = ann.radius * vs.scale;
          ctx.beginPath();
          ctx.arc(center.cx, center.cy, rCanvas, 0, Math.PI * 2);
          ctx.fillStyle = hexToRgba(ann.color, ANNOTATION_ALPHA);
          ctx.fill();
          ctx.strokeStyle = ann.color;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
          if (isSelected) {
            // Center handle
            ctx.fillStyle = ann.color;
            ctx.fillRect(center.cx - 3, center.cy - 3, 6, 6);
            // Radius resize handle at the right edge (3 o'clock)
            const rh = imageToCanvas(ann.centerX + ann.radius, ann.centerY, vs);
            ctx.strokeStyle = ann.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(rh.cx, rh.cy, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.stroke();
          }
          break;
        }
        case 'freehand': {
          if (ann.points.length < 2) break;
          ctx.beginPath();
          const f0 = imageToCanvas(ann.points[0].x, ann.points[0].y, vs);
          ctx.moveTo(f0.cx, f0.cy);
          for (let i = 1; i < ann.points.length; i++) {
            const fp = imageToCanvas(ann.points[i].x, ann.points[i].y, vs);
            ctx.lineTo(fp.cx, fp.cy);
          }
          ctx.closePath();
          ctx.fillStyle = hexToRgba(ann.color, ANNOTATION_ALPHA);
          ctx.fill();
          ctx.strokeStyle = ann.color;
          ctx.lineWidth = lineWidth;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.stroke();
          break;
        }
        case 'point': {
          const pp = imageToCanvas(ann.x, ann.y, vs);
          // Crosshair
          const size = POINT_MARKER_RADIUS + 2;
          ctx.strokeStyle = ann.color;
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          ctx.beginPath();
          ctx.moveTo(pp.cx - size, pp.cy);
          ctx.lineTo(pp.cx + size, pp.cy);
          ctx.moveTo(pp.cx, pp.cy - size);
          ctx.lineTo(pp.cx, pp.cy + size);
          ctx.stroke();
          // Dot
          ctx.beginPath();
          ctx.arc(pp.cx, pp.cy, 3, 0, Math.PI * 2);
          ctx.fillStyle = ann.color;
          ctx.fill();
          break;
        }
        case 'rect': {
          const tl = imageToCanvas(ann.x, ann.y, vs);
          const rw = ann.width * vs.scale;
          const rh = ann.height * vs.scale;
          ctx.beginPath();
          ctx.rect(tl.cx, tl.cy, rw, rh);
          ctx.fillStyle = hexToRgba(ann.color, ANNOTATION_ALPHA);
          ctx.fill();
          ctx.strokeStyle = ann.color;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
          if (isSelected) {
            // Corner handles
            for (const [hx, hy] of [[ann.x, ann.y], [ann.x + ann.width, ann.y], [ann.x + ann.width, ann.y + ann.height], [ann.x, ann.y + ann.height]]) {
              const h = imageToCanvas(hx, hy, vs);
              ctx.fillStyle = ann.color;
              ctx.fillRect(h.cx - 3, h.cy - 3, 6, 6);
            }
          }
          break;
        }
      }

      // Label (only if labels visible)
      if (ann.label && labelsVisible) {
        let lx: number, ly: number;
        switch (ann.type) {
          case 'polygon':
          case 'freehand': {
            const minX = Math.min(...ann.points.map((p) => p.x));
            const minY = Math.min(...ann.points.map((p) => p.y));
            const c = imageToCanvas(minX, minY, vs);
            lx = c.cx;
            ly = c.cy - 6;
            break;
          }
          case 'circle': {
            const ct = imageToCanvas(ann.centerX, ann.centerY - ann.radius, vs);
            lx = ct.cx;
            ly = ct.cy - 6;
            break;
          }
          case 'rect': {
            const tl = imageToCanvas(ann.x, ann.y, vs);
            lx = tl.cx;
            ly = tl.cy - 6;
            break;
          }
          case 'point': {
            const pt = imageToCanvas(ann.x, ann.y, vs);
            lx = pt.cx + POINT_MARKER_RADIUS + 4;
            ly = pt.cy - 4;
            break;
          }
          default:
            return;
        }
        // Scale-aware label sizing: base 12px in image-space, clamped to [9, 14] screen-px
        // so labels shrink with zoom-out (avoiding occlusion) but stay readable.
        const baseFontPx = isSelected ? 13 : 12;
        const scaledFont = Math.min(14, Math.max(9, baseFontPx * vs.scale));
        const fontWeight = isSelected ? 'bold ' : '';
        ctx.font = `${fontWeight}${scaledFont.toFixed(1)}px Arial`;
        const tw = ctx.measureText(ann.label).width + 8;
        const th = scaledFont + 4;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(lx, ly - th + 2, tw, th);
        ctx.fillStyle = '#fff';
        ctx.fillText(ann.label, lx + 4, ly);
      }
    },
    [imageToCanvas],
  );

  // --- Draw live (in-progress) shape ---
  const drawLiveShape = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      dr: typeof drawingRef.current,
      vs: ViewState,
    ) => {
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;

      if (dr.tool === 'polygon' && dr.points.length > 0) {
        ctx.strokeStyle = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        ctx.fillStyle = hexToRgba(
          ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length],
          0.15,
        );
        ctx.beginPath();
        const p0 = imageToCanvas(dr.points[0].x, dr.points[0].y, vs);
        ctx.moveTo(p0.cx, p0.cy);
        for (let i = 1; i < dr.points.length; i++) {
          const p = imageToCanvas(dr.points[i].x, dr.points[i].y, vs);
          ctx.lineTo(p.cx, p.cy);
        }
        // Live rubber-band to mouse
        if (dr.currentMouse) {
          const pm = imageToCanvas(dr.currentMouse.x, dr.currentMouse.y, vs);
          ctx.lineTo(pm.cx, pm.cy);
        }
        ctx.stroke();
        // Vertex dots
        for (const pt of dr.points) {
          const c = imageToCanvas(pt.x, pt.y, vs);
          ctx.fillStyle = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
          ctx.setLineDash([]);
          ctx.fillRect(c.cx - 3, c.cy - 3, 6, 6);
          ctx.setLineDash([5, 5]);
        }
      }

      if (dr.tool === 'circle' && dr.startPoint) {
        const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        const center = imageToCanvas(dr.startPoint.x, dr.startPoint.y, vs);
        const rCanvas = dr.currentRadius * vs.scale;
        ctx.beginPath();
        ctx.arc(center.cx, center.cy, rCanvas, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.fillStyle = hexToRgba(color, 0.15);
        ctx.fill();
      }

      if (dr.tool === 'freehand' && dr.points.length > 1) {
        const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        ctx.strokeStyle = color;
        ctx.beginPath();
        const f0 = imageToCanvas(dr.points[0].x, dr.points[0].y, vs);
        ctx.moveTo(f0.cx, f0.cy);
        for (let i = 1; i < dr.points.length; i++) {
          const fp = imageToCanvas(dr.points[i].x, dr.points[i].y, vs);
          ctx.lineTo(fp.cx, fp.cy);
        }
        ctx.stroke();
      }

      if (dr.tool === 'rect' && dr.startPoint && dr.currentMouse) {
        const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        const p0 = imageToCanvas(dr.startPoint.x, dr.startPoint.y, vs);
        const p1 = imageToCanvas(dr.currentMouse.x, dr.currentMouse.y, vs);
        const rx = Math.min(p0.cx, p1.cx);
        const ry = Math.min(p0.cy, p1.cy);
        const rw = Math.abs(p1.cx - p0.cx);
        const rh = Math.abs(p1.cy - p0.cy);
        ctx.strokeStyle = color;
        ctx.fillStyle = hexToRgba(color, 0.15);
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillRect(rx, ry, rw, rh);
      }

      ctx.setLineDash([]);
    },
    [annotations.length, imageToCanvas],
  );

  // --- Request animation frame loop ---
  useEffect(() => {
    const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
    if (!canvas) return;
    syncCanvasSize(canvas);
    renderCanvas(canvas);
  }, [fullscreen, viewState, annotations, selectedIdx, imgDimensions, renderCanvas, syncCanvasSize]);

  // Refit when entering/leaving fullscreen or image changes
  useEffect(() => {
    const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
    // Small delay to let DOM update
    const timer = setTimeout(() => {
      syncCanvasSize(canvas);
      fitImageToCanvas(canvas);
    }, 50);
    return () => clearTimeout(timer);
  }, [fullscreen, imgDimensions, fitImageToCanvas, syncCanvasSize]);

  // --- API: Fetch slice ---
  const fetchSlice = useCallback(
    async (sessionId: string, sliceIdx: number, view: string) => {
      const fetchId = ++fetchRef.current;
      setLoading(true);
      try {
        const res = await api.getSlice(sessionId, sliceIdx, view);
        if (fetchRef.current !== fetchId) return;
        updateNodeData(id, {
          status: 'success',
          error: undefined,
          imageBase64: res.image_base64,
          sliceIndex: res.slice_index,
          totalSlices: res.total_slices,
        } as Partial<InteractiveAnnotatorNodeData>);
      } catch (err) {
        if (fetchRef.current !== fetchId) return;
        updateNodeData(id, {
          status: 'error',
          error: err instanceof Error ? err.message : 'Failed to fetch slice',
        } as Partial<InteractiveAnnotatorNodeData>);
      } finally {
        if (fetchRef.current === fetchId) setLoading(false);
      }
    },
    [id, updateNodeData],
  );

  // Auto-fetch when sessionId becomes available
  useEffect(() => {
    if (d.sessionId && !d.imageBase64) {
      fetchSlice(d.sessionId, d.sliceIndex || 0, d.view || 'axial');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d.sessionId]);

  // --- Navigation (with per-slice annotation persistence) ---
  const goPrev = useCallback(() => {
    if (!d.sessionId || d.sliceIndex <= 0) return;
    const next = d.sliceIndex - 1;
    // Save current slice annotations
    const map = { ...(d.sliceAnnotationsMap || {}) };
    map[d.sliceIndex] = annotations;
    // Restore target slice annotations
    const targetAnns = map[next] || [];
    updateNodeData(id, {
      sliceIndex: next,
      annotations: targetAnns,
      sliceAnnotationsMap: map,
    } as Partial<InteractiveAnnotatorNodeData>);
    fetchSlice(d.sessionId, next, d.view || 'axial');
  }, [d.sessionId, d.sliceIndex, d.view, d.sliceAnnotationsMap, annotations, id, updateNodeData, fetchSlice]);

  const goNext = useCallback(() => {
    if (!d.sessionId || d.sliceIndex >= d.totalSlices - 1) return;
    const next = d.sliceIndex + 1;
    const map = { ...(d.sliceAnnotationsMap || {}) };
    map[d.sliceIndex] = annotations;
    const targetAnns = map[next] || [];
    updateNodeData(id, {
      sliceIndex: next,
      annotations: targetAnns,
      sliceAnnotationsMap: map,
    } as Partial<InteractiveAnnotatorNodeData>);
    fetchSlice(d.sessionId, next, d.view || 'axial');
  }, [d.sessionId, d.sliceIndex, d.totalSlices, d.view, d.sliceAnnotationsMap, annotations, id, updateNodeData, fetchSlice]);

  const handleSlider = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!d.sessionId) return;
      const idx = parseInt(e.target.value, 10);
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[d.sliceIndex] = annotations;
      const targetAnns = map[idx] || [];
      updateNodeData(id, {
        sliceIndex: idx,
        annotations: targetAnns,
        sliceAnnotationsMap: map,
      } as Partial<InteractiveAnnotatorNodeData>);
      fetchSlice(d.sessionId, idx, d.view || 'axial');
    },
    [d.sessionId, d.view, d.sliceAnnotationsMap, d.sliceIndex, annotations, id, updateNodeData, fetchSlice],
  );

  const handleViewChange = useCallback(
    (view: 'axial' | 'coronal' | 'sagittal') => {
      if (!d.sessionId) return;
      // Save current annotations before view change
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[d.sliceIndex] = annotations;
      // New view starts fresh (different orientation = different slice set)
      updateNodeData(id, {
        view,
        sliceIndex: 0,
        annotations: [],
        sliceAnnotationsMap: {},
      } as Partial<InteractiveAnnotatorNodeData>);
      fetchSlice(d.sessionId, 0, view);
    },
    [d.sessionId, d.sliceAnnotationsMap, d.sliceIndex, annotations, id, updateNodeData, fetchSlice],
  );

  // --- Current-slice SAM2 segmentation ---
  const handleRunAutoSegmentation = useCallback(async () => {
    if (!d.sessionId) return;

    setSegRunning(true);
    setSegMessage(null);

    try {
      const configName = connectedAutoSeg?.configName || 'fast';
      const res = await api.autoSegment(
        d.sessionId,
        d.sliceIndex,
        d.view || 'axial',
        configName,
      );

      // Log full API response for debugging
      console.log('[AutoSeg] Response:', {
        count: res.count,
        raw_mask_count: res.raw_mask_count,
        config_used: res.config_used,
        elapsed_seconds: res.elapsed_seconds,
        message: res.message,
        shapesSample: res.shapes.slice(0, 2),
      });

      // Convert API shapes to AnnotationShape polygons
      const newShapes: AnnotationShape[] = res.shapes.map((s) => ({
        type: 'polygon' as const,
        points: s.points.map((p) => ({ x: p.x, y: p.y })),
        label: s.label,
        color: s.color,
      }));

      // Merge with existing annotations on this slice
      pushHistory(annotations);
      const merged = [...annotations, ...newShapes];
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[d.sliceIndex] = merged;

      updateNodeData(id, {
        annotations: merged,
        sliceAnnotationsMap: map,
        activeTool: 'pan',
      } as Partial<InteractiveAnnotatorNodeData>);

      setSegMessage(
        `Done: ${res.count} ROI${res.count === 1 ? '' : 's'} (${res.raw_mask_count} raw, ${res.raw_mask_count - res.count} filtered) in ${res.elapsed_seconds.toFixed(1)}s using ${res.config_used}.`,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setSegMessage(`Error: ${msg}`);
    } finally {
      setSegRunning(false);
    }
  }, [connectedAutoSeg, d.sessionId, d.sliceIndex, d.view, d.sliceAnnotationsMap, annotations, id, updateNodeData, pushHistory]);

  const handleRunPromptSegmentation = useCallback(async () => {
    if (!d.sessionId || !activePrompt) {
      setSegMessage('Error: draw or select a point/rectangle prompt first.');
      return;
    }

    setSegRunning(true);
    setSegMessage(null);

    try {
      const configName = connectedAutoSeg?.configName || 'fast';
      const res = await api.promptSegment(
        d.sessionId,
        d.sliceIndex,
        d.view || 'axial',
        activePrompt.points,
        activePrompt.boxes,
        configName,
      );

      const newShapes: AnnotationShape[] = res.shapes.map((s) => ({
        type: 'polygon' as const,
        points: s.points.map((p) => ({ x: p.x, y: p.y })),
        label: s.label,
        color: s.color,
      }));

      pushHistory(annotations);
      const merged = [...annotations, ...newShapes];
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[d.sliceIndex] = merged;

      updateNodeData(id, {
        annotations: merged,
        sliceAnnotationsMap: map,
        activeTool: 'pan',
        segmentationPrompt: activePrompt,
      } as Partial<InteractiveAnnotatorNodeData>);

      setSegMessage(
        `Done: prompt SAM2 added ${res.count} ROI${res.count === 1 ? '' : 's'} from ${activePrompt.points.length} point(s) and ${activePrompt.boxes.length} box(es) in ${res.elapsed_seconds.toFixed(1)}s.`,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setSegMessage(`Error: ${msg}`);
    } finally {
      setSegRunning(false);
    }
  }, [activePrompt, annotations, connectedAutoSeg, d.sessionId, d.sliceAnnotationsMap, d.sliceIndex, d.view, id, pushHistory, updateNodeData]);

  // --- Tool selection ---
  const setTool = useCallback(
    (tool: AnnotationTool) => {
      // Cancel any in-progress drawing
      drawingRef.current = {
        active: false,
        tool,
        points: [],
        startPoint: null,
        currentRadius: 0,
        currentMouse: null,
      };
      setSelectedIdx(-1);
      updateNodeData(id, { activeTool: tool } as Partial<InteractiveAnnotatorNodeData>);
    },
    [id, updateNodeData],
  );

  // --- Zoom ---
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const delta = e.deltaY > 0 ? 1 / ZOOM_FACTOR : ZOOM_FACTOR;
      const newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, viewState.scale * delta));

      // Zoom towards mouse position (same math as Gradio's handleMouseWheel)
      const worldX = (mouseX - viewState.offsetX) / viewState.scale;
      const worldY = (mouseY - viewState.offsetY) / viewState.scale;
      const newOffsetX = mouseX - worldX * newScale;
      const newOffsetY = mouseY - worldY * newScale;

      setViewState({ scale: newScale, offsetX: newOffsetX, offsetY: newOffsetY });
    },
    [fullscreen, viewState],
  );

  const zoomIn = useCallback(() => {
    const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
    if (!canvas) return;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const newScale = Math.min(MAX_ZOOM, viewState.scale * ZOOM_FACTOR);
    const worldX = (cx - viewState.offsetX) / viewState.scale;
    const worldY = (cy - viewState.offsetY) / viewState.scale;
    setViewState({ scale: newScale, offsetX: cx - worldX * newScale, offsetY: cy - worldY * newScale });
  }, [fullscreen, viewState]);

  const zoomOut = useCallback(() => {
    const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
    if (!canvas) return;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const newScale = Math.max(MIN_ZOOM, viewState.scale / ZOOM_FACTOR);
    const worldX = (cx - viewState.offsetX) / viewState.scale;
    const worldY = (cy - viewState.offsetY) / viewState.scale;
    setViewState({ scale: newScale, offsetX: cx - worldX * newScale, offsetY: cy - worldY * newScale });
  }, [fullscreen, viewState]);

  const zoomReset = useCallback(() => {
    const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
    fitImageToCanvas(canvas);
  }, [fullscreen, fitImageToCanvas]);

  // --- Mouse event: get canvas-relative coords ---
  const getCanvasCoords = useCallback(
    (e: React.MouseEvent): { cx: number; cy: number } | null => {
      const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      return { cx: e.clientX - rect.left, cy: e.clientY - rect.top };
    },
    [fullscreen],
  );

  // --- Hit test: which annotation is under cursor? ---
  const hitTestAnnotation = useCallback(
    (imgPt: ImagePoint): number => {
      // Check in reverse order (top-most first)
      for (let i = annotations.length - 1; i >= 0; i--) {
        const ann = annotations[i];
        switch (ann.type) {
          case 'point': {
            const dist = Math.sqrt((imgPt.x - ann.x) ** 2 + (imgPt.y - ann.y) ** 2);
            if (dist < POINT_MARKER_RADIUS / viewState.scale + 5) return i;
            break;
          }
          case 'circle': {
            const dist = Math.sqrt((imgPt.x - ann.centerX) ** 2 + (imgPt.y - ann.centerY) ** 2);
            if (dist <= ann.radius) return i;
            break;
          }
          case 'rect': {
            if (
              imgPt.x >= ann.x && imgPt.x <= ann.x + ann.width &&
              imgPt.y >= ann.y && imgPt.y <= ann.y + ann.height
            ) return i;
            break;
          }
          case 'polygon':
          case 'freehand': {
            // Ray-casting point-in-polygon
            const pts = ann.points;
            if (pts.length < 3) break;
            let inside = false;
            for (let j = 0, k = pts.length - 1; j < pts.length; k = j++) {
              const xi = pts[j].x, yi = pts[j].y;
              const xk = pts[k].x, yk = pts[k].y;
              if ((yi > imgPt.y) !== (yk > imgPt.y) &&
                imgPt.x < ((xk - xi) * (imgPt.y - yi)) / (yk - yi) + xi) {
                inside = !inside;
              }
            }
            if (inside) return i;
            break;
          }
        }
      }
      return -1;
    },
    [annotations, viewState.scale],
  );

  // Eraser active painting flag
  const eraserActiveRef = useRef(false);
  // Snapshot taken at start of eraser stroke (for single undo per stroke)
  const eraserHistorySnapshotRef = useRef<AnnotationShape[] | null>(null);

  // --- Helper: commit a finished annotation (stages it and opens label editor) ---
  const commitAnnotation = useCallback(
    (shape: AnnotationShape) => {
      // Stage the shape and open the label editor popup
      setLabelEditor({ open: true, mode: 'new', pendingShape: shape, editIdx: -1 });
      // Auto-switch to pan mode after completing a shape
      drawingRef.current = {
        active: false,
        tool: 'pan',
        points: [],
        startPoint: null,
        currentRadius: 0,
        currentMouse: null,
      };
      updateNodeData(id, { activeTool: 'pan' } as Partial<InteractiveAnnotatorNodeData>);
    },
    [id, updateNodeData],
  );

  // --- Label editor: confirm (new shape or edit existing) ---
  const handleLabelConfirm = useCallback(
    (label: string, color: string) => {
      if (labelEditor.mode === 'new' && labelEditor.pendingShape) {
        const shape = { ...labelEditor.pendingShape, label, color };
        pushHistory(annotations);
        const newAnnotations = [...annotations, shape];
        // Persist to sliceAnnotationsMap
        const map = { ...(d.sliceAnnotationsMap || {}) };
        map[d.sliceIndex] = newAnnotations;
        updateNodeData(id, {
          annotations: newAnnotations,
          sliceAnnotationsMap: map,
        } as Partial<InteractiveAnnotatorNodeData>);
      } else if (labelEditor.mode === 'edit' && labelEditor.editIdx >= 0) {
        pushHistory(annotations);
        const newAnnotations = [...annotations];
        newAnnotations[labelEditor.editIdx] = { ...newAnnotations[labelEditor.editIdx], label, color };
        const map = { ...(d.sliceAnnotationsMap || {}) };
        map[d.sliceIndex] = newAnnotations;
        updateNodeData(id, {
          annotations: newAnnotations,
          sliceAnnotationsMap: map,
        } as Partial<InteractiveAnnotatorNodeData>);
      }
      setLabelEditor({ open: false, mode: 'new', pendingShape: null, editIdx: -1 });
    },
    [labelEditor, annotations, d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, pushHistory],
  );

  // --- Label editor: cancel ---
  const handleLabelCancel = useCallback(() => {
    setLabelEditor({ open: false, mode: 'new', pendingShape: null, editIdx: -1 });
  }, []);

  // --- Delete selected annotation ---
  const deleteSelected = useCallback(() => {
    if (selectedIdx < 0 || selectedIdx >= annotations.length) return;
    pushHistory(annotations);
    const newAnnotations = annotations.filter((_, i) => i !== selectedIdx);
    const map = { ...(d.sliceAnnotationsMap || {}) };
    map[d.sliceIndex] = newAnnotations;
    updateNodeData(id, {
      annotations: newAnnotations,
      sliceAnnotationsMap: map,
    } as Partial<InteractiveAnnotatorNodeData>);
    setSelectedIdx(-1);
  }, [selectedIdx, annotations, d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, pushHistory]);

  const clearAll = useCallback(() => {
    pushHistory(annotations);
    const map = { ...(d.sliceAnnotationsMap || {}) };
    map[d.sliceIndex] = [];
    updateNodeData(id, {
      annotations: [],
      sliceAnnotationsMap: map,
    } as Partial<InteractiveAnnotatorNodeData>);
    setSelectedIdx(-1);
  }, [d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, annotations, pushHistory]);

  // --- Download current slice with annotations as PNG ---
  const downloadSlice = useCallback(() => {
    const img = imageObjRef.current;
    if (!img || !imgDimensions) return;
    const offscreen = document.createElement('canvas');
    offscreen.width = imgDimensions.w;
    offscreen.height = imgDimensions.h;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;
    // Draw the base image at native resolution
    ctx.drawImage(img, 0, 0, imgDimensions.w, imgDimensions.h);
    // Draw all annotations at scale=1 (image coords == canvas coords)
    const vs1: ViewState = { scale: 1, offsetX: 0, offsetY: 0 };
    for (let i = 0; i < annotations.length; i++) {
      drawAnnotation(ctx, annotations[i], vs1, i === selectedIdx, showLabels);
    }
    offscreen.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `slice_${(d.sliceIndex ?? 0) + 1}_${currentView}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }, [imageObjRef, imgDimensions, annotations, selectedIdx, showLabels, drawAnnotation, d.sliceIndex, currentView]);

  // --- Pointer down ---
  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      const coords = getCanvasCoords(e);
      if (!coords) return;
      const imgPt = canvasToImage(coords.cx, coords.cy, viewState);

      // Middle-button → always pan
      if (e.button === 1) {
        panRef.current = {
          active: true,
          startX: e.clientX,
          startY: e.clientY,
          startOX: viewState.offsetX,
          startOY: viewState.offsetY,
        };
        return;
      }

      // Pan tool → 1) vertex drag on selected shape, 2) shape drag, 3) pan
      if (currentTool === 'pan') {
        if (e.button !== 0) return; // ignore right-click / middle-click in pan mode
        // 1. Vertex hit on selected annotation
        if (selectedIdx >= 0 && selectedIdx < annotations.length) {
          const sel = annotations[selectedIdx];
          const VR = 8 / viewState.scale;
          let foundVtx = -1;
          switch (sel.type) {
            case 'polygon':
            case 'freehand':
              for (let vi = 0; vi < sel.points.length; vi++) {
                const dx = imgPt.x - sel.points[vi].x;
                const dy = imgPt.y - sel.points[vi].y;
                if (Math.sqrt(dx * dx + dy * dy) < VR) { foundVtx = vi; break; }
              }
              break;
            case 'rect': {
              const corners = [
                { x: sel.x,             y: sel.y },
                { x: sel.x + sel.width, y: sel.y },
                { x: sel.x + sel.width, y: sel.y + sel.height },
                { x: sel.x,             y: sel.y + sel.height },
              ];
              for (let vi = 0; vi < corners.length; vi++) {
                const dx = imgPt.x - corners[vi].x;
                const dy = imgPt.y - corners[vi].y;
                if (Math.sqrt(dx * dx + dy * dy) < VR) { foundVtx = vi; break; }
              }
              break;
            }
            case 'circle': {
              const dx = imgPt.x - (sel.centerX + sel.radius);
              const dy = imgPt.y - sel.centerY;
              if (Math.sqrt(dx * dx + dy * dy) < VR) foundVtx = 0;
              break;
            }
            default:
              break;
          }
          if (foundVtx >= 0) {
            vertexDragRef.current = {
              active: true,
              annIdx: selectedIdx,
              vertexIdx: foundVtx,
              originalAnn: JSON.parse(JSON.stringify(sel)),
            };
            return;
          }
        }
        // 2. Shape drag
        const hitIdx = hitTestAnnotation(imgPt);
        if (hitIdx >= 0) {
          setSelectedIdx(hitIdx);
          dragRef.current = {
            active: true,
            annIdx: hitIdx,
            startImg: { x: imgPt.x, y: imgPt.y },
            originalAnn: JSON.parse(JSON.stringify(annotations[hitIdx])),
          };
          return;
        }
        // 3. Pan
        panRef.current = {
          active: true,
          startX: e.clientX,
          startY: e.clientY,
          startOX: viewState.offsetX,
          startOY: viewState.offsetY,
        };
        return;
      }

      if (e.button !== 0) return; // only left-click for drawing

      // Eraser tool: start stroke, take history snapshot, immediately erase at click point
      if (currentTool === 'eraser') {
        eraserActiveRef.current = true;
        eraserHistorySnapshotRef.current = JSON.parse(JSON.stringify(annotations));
        // Clip annotations by eraser geometry at click point
        let eraserChanged0 = false;
        const erased0 = annotations
          .map((ann) => { const r = eraseShapeByCircle(ann, imgPt, eraserRadius); if (r !== ann) eraserChanged0 = true; return r; })
          .filter((ann): ann is AnnotationShape => ann !== null);
        if (eraserChanged0) {
          const map = { ...(d.sliceAnnotationsMap || {}) };
          map[d.sliceIndex] = erased0;
          updateNodeData(id, { annotations: erased0, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
          setSelectedIdx(-1);
        }
        return;
      }

      switch (currentTool) {
        case 'rect': {
          drawingRef.current = {
            active: true,
            tool: 'rect',
            points: [],
            startPoint: imgPt,
            currentRadius: 0,
            currentMouse: imgPt,
          };
          break;
        }
        case 'point': {
          const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
          commitAnnotation({
            type: 'point',
            x: imgPt.x,
            y: imgPt.y,
            label: `P${annotations.length + 1}`,
            color,
          });
          break;
        }
        case 'polygon': {
          const dr = drawingRef.current;
          if (!dr.active) {
            // Start new polygon
            dr.active = true;
            dr.tool = 'polygon';
            dr.points = [imgPt];
            dr.currentMouse = imgPt;
          } else {
            // Check if clicking near start point to close
            if (dr.points.length >= 3) {
              const first = dr.points[0];
              const distToStart = Math.sqrt((imgPt.x - first.x) ** 2 + (imgPt.y - first.y) ** 2);
              if (distToStart < 10 / viewState.scale) {
                // Close polygon
                const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
                commitAnnotation({
                  type: 'polygon',
                  points: [...dr.points],
                  label: `Poly${annotations.length + 1}`,
                  color,
                });
                dr.active = false;
                dr.points = [];
                dr.currentMouse = null;
                break;
              }
            }
            dr.points.push(imgPt);
          }
          // Force re-render for live preview
          requestRender();
          break;
        }
        case 'circle': {
          drawingRef.current = {
            active: true,
            tool: 'circle',
            points: [],
            startPoint: imgPt,
            currentRadius: 0,
            currentMouse: null,
          };
          break;
        }
        case 'freehand': {
          drawingRef.current = {
            active: true,
            tool: 'freehand',
            points: [imgPt],
            startPoint: null,
            currentRadius: 0,
            currentMouse: null,
          };
          break;
        }
      }
    },
    [currentTool, viewState, getCanvasCoords, canvasToImage, hitTestAnnotation, commitAnnotation, annotations, d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, eraserRadius],
  );

  // --- Pointer move ---
  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      // Eraser: update cursor preview always; erase while button held
      if (currentTool === 'eraser') {
        const coords = getCanvasCoords(e);
        if (coords) {
          const imgPt = canvasToImage(coords.cx, coords.cy, viewState);
          eraserCursorRef.current = imgPt;
          requestRender();
          if (eraserActiveRef.current) {
            let eraserChanged1 = false;
            const erased1 = annotations
              .map((ann) => { const r = eraseShapeByCircle(ann, imgPt, eraserRadius); if (r !== ann) eraserChanged1 = true; return r; })
              .filter((ann): ann is AnnotationShape => ann !== null);
            if (eraserChanged1) {
              const map = { ...(d.sliceAnnotationsMap || {}) };
              map[d.sliceIndex] = erased1;
              updateNodeData(id, { annotations: erased1, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
              setSelectedIdx(-1);
            }
          }
        }
        return;
      }

      // Vertex drag (reshape)
      if (vertexDragRef.current.active) {
        const vCoords = getCanvasCoords(e);
        if (!vCoords) return;
        const vImgPt = canvasToImage(vCoords.cx, vCoords.cy, viewState);
        const { annIdx, vertexIdx } = vertexDragRef.current;
        const ann = annotations[annIdx];
        if (!ann) return;
        let updated: AnnotationShape;
        switch (ann.type) {
          case 'polygon':
          case 'freehand': {
            const pts = [...ann.points];
            pts[vertexIdx] = { x: vImgPt.x, y: vImgPt.y };
            updated = { ...ann, points: pts };
            break;
          }
          case 'rect': {
            // corners: 0=TL,1=TR,2=BR,3=BL — opposite corner is (idx+2)%4
            const oppCorners = [
              { x: ann.x + ann.width,  y: ann.y + ann.height }, // 0 → opp is 2 (BR)
              { x: ann.x,              y: ann.y + ann.height }, // 1 → opp is 3 (BL)
              { x: ann.x,              y: ann.y },              // 2 → opp is 0 (TL)
              { x: ann.x + ann.width,  y: ann.y },              // 3 → opp is 1 (TR)
            ];
            const opp = oppCorners[vertexIdx];
            updated = {
              ...ann,
              x: Math.min(vImgPt.x, opp.x),
              y: Math.min(vImgPt.y, opp.y),
              width: Math.abs(vImgPt.x - opp.x),
              height: Math.abs(vImgPt.y - opp.y),
            };
            break;
          }
          case 'circle': {
            const dx = vImgPt.x - ann.centerX;
            const dy = vImgPt.y - ann.centerY;
            updated = { ...ann, radius: Math.max(1, Math.sqrt(dx * dx + dy * dy)) };
            break;
          }
          default:
            return;
        }
        const vAnns = [...annotations];
        vAnns[annIdx] = updated;
        const vMap = { ...(d.sliceAnnotationsMap || {}) };
        vMap[d.sliceIndex] = vAnns;
        updateNodeData(id, { annotations: vAnns, sliceAnnotationsMap: vMap } as Partial<InteractiveAnnotatorNodeData>);
        return;
      }

      // Drag annotation
      if (dragRef.current.active) {
        const dCoords = getCanvasCoords(e);
        if (!dCoords) return;
        const dImgPt = canvasToImage(dCoords.cx, dCoords.cy, viewState);
        const dx = dImgPt.x - dragRef.current.startImg.x;
        const dy = dImgPt.y - dragRef.current.startImg.y;
        const orig = dragRef.current.originalAnn;
        if (!orig) return;
        const idx = dragRef.current.annIdx;
        let moved: AnnotationShape;
        switch (orig.type) {
          case 'polygon':
            moved = { ...orig, points: orig.points.map((p) => ({ x: p.x + dx, y: p.y + dy })) };
            break;
          case 'freehand':
            moved = { ...orig, points: orig.points.map((p) => ({ x: p.x + dx, y: p.y + dy })) };
            break;
          case 'circle':
            moved = { ...orig, centerX: orig.centerX + dx, centerY: orig.centerY + dy };
            break;
          case 'rect':
            moved = { ...orig, x: orig.x + dx, y: orig.y + dy };
            break;
          case 'point':
            moved = { ...orig, x: orig.x + dx, y: orig.y + dy };
            break;
          default:
            return;
        }
        const newAnns = [...annotations];
        newAnns[idx] = moved;
        const map = { ...(d.sliceAnnotationsMap || {}) };
        map[d.sliceIndex] = newAnns;
        updateNodeData(id, {
          annotations: newAnns,
          sliceAnnotationsMap: map,
        } as Partial<InteractiveAnnotatorNodeData>);
        return;
      }

      // Pan
      if (panRef.current.active) {
        const dx = e.clientX - panRef.current.startX;
        const dy = e.clientY - panRef.current.startY;
        setViewState((prev) => ({
          ...prev,
          offsetX: panRef.current.startOX + dx,
          offsetY: panRef.current.startOY + dy,
        }));
        return;
      }

      const coords = getCanvasCoords(e);
      if (!coords) return;
      const imgPt = canvasToImage(coords.cx, coords.cy, viewState);

      const dr = drawingRef.current;
      if (!dr.active) return;

      if (dr.tool === 'polygon') {
        dr.currentMouse = imgPt;
        requestRender();
      }

      if (dr.tool === 'rect') {
        dr.currentMouse = imgPt;
        requestRender();
      }

      if (dr.tool === 'circle' && dr.startPoint) {
        const dx = imgPt.x - dr.startPoint.x;
        const dy = imgPt.y - dr.startPoint.y;
        dr.currentRadius = Math.sqrt(dx * dx + dy * dy);
        requestRender();
      }

      if (dr.tool === 'freehand') {
        const lastPt = dr.points[dr.points.length - 1];
        const dist = Math.sqrt((imgPt.x - lastPt.x) ** 2 + (imgPt.y - lastPt.y) ** 2);
        if (dist > MIN_FREEHAND_DIST) {
          dr.points.push(imgPt);
          requestRender();
        }
      }
    },
    [viewState, getCanvasCoords, canvasToImage, annotations, d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, currentTool, eraserRadius],
  );

  // --- Pointer up ---
  const handlePointerUp = useCallback(
    () => {
      // Eraser stroke end: push history snapshot taken at stroke start
      if (eraserActiveRef.current) {
        eraserActiveRef.current = false;
        if (eraserHistorySnapshotRef.current !== null) {
          undoStackRef.current[d.sliceIndex] = undoStackRef.current[d.sliceIndex] || [];
          undoStackRef.current[d.sliceIndex].push(eraserHistorySnapshotRef.current);
          if (undoStackRef.current[d.sliceIndex].length > 50) undoStackRef.current[d.sliceIndex].shift();
          if (!redoStackRef.current[d.sliceIndex]) redoStackRef.current[d.sliceIndex] = [];
          redoStackRef.current[d.sliceIndex] = [];
          eraserHistorySnapshotRef.current = null;
        }
        // Switch to pan tool after completing an eraser stroke
        updateNodeData(id, { activeTool: 'pan' } as Partial<InteractiveAnnotatorNodeData>);
        return;
      }

      // Stop vertex drag
      if (vertexDragRef.current.active) {
        if (vertexDragRef.current.originalAnn !== null) {
          const origAnns = [...annotations];
          origAnns[vertexDragRef.current.annIdx] = vertexDragRef.current.originalAnn;
          pushHistory(origAnns);
        }
        vertexDragRef.current.active = false;
        vertexDragRef.current.originalAnn = null;
        return;
      }

      // Stop dragging annotation
      if (dragRef.current.active) {
        if (dragRef.current.originalAnn !== null) {
          const origAnns = [...annotations];
          origAnns[dragRef.current.annIdx] = dragRef.current.originalAnn;
          pushHistory(origAnns);
        }
        dragRef.current.active = false;
        dragRef.current.originalAnn = null;
        return;
      }

      // Stop panning
      if (panRef.current.active) {
        panRef.current.active = false;
        return;
      }

      const dr = drawingRef.current;
      if (!dr.active) return;

      // Polygon: don't finish on mouseup (it's click-to-place)
      if (dr.tool === 'polygon') return;

      if (dr.tool === 'rect' && dr.startPoint && dr.currentMouse) {
        const x = Math.min(dr.startPoint.x, dr.currentMouse.x);
        const y = Math.min(dr.startPoint.y, dr.currentMouse.y);
        const width = Math.abs(dr.currentMouse.x - dr.startPoint.x);
        const height = Math.abs(dr.currentMouse.y - dr.startPoint.y);
        if (width > 3 && height > 3) {
          const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
          commitAnnotation({
            type: 'rect',
            x,
            y,
            width,
            height,
            label: `R${annotations.length + 1}`,
            color,
          });
        }
      }

      if (dr.tool === 'circle' && dr.startPoint && dr.currentRadius > 3) {
        const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        commitAnnotation({
          type: 'circle',
          centerX: dr.startPoint.x,
          centerY: dr.startPoint.y,
          radius: dr.currentRadius,
          label: `C${annotations.length + 1}`,
          color,
        });
      }

      if (dr.tool === 'freehand' && dr.points.length > 2) {
        const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
        commitAnnotation({
          type: 'freehand',
          points: [...dr.points],
          label: `FH${annotations.length + 1}`,
          color,
        });
      }

      // Reset drawing state (polygon returns early above, so this is always non-polygon)
      dr.active = false;
      dr.points = [];
      dr.startPoint = null;
      dr.currentRadius = 0;
      dr.currentMouse = null;
    },
    [annotations.length, commitAnnotation, pushHistory, d.sliceIndex, id, updateNodeData],
  );

  // Clear eraser cursor when pointer leaves canvas
  const handlePointerLeave = useCallback(() => {
    if (currentTool === 'eraser') {
      eraserCursorRef.current = null;
      // requestRender is called via the animation frame loop triggered by
      // the eraser cursor ref change — we use a direct rAF call here instead
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = requestAnimationFrame(() => {
        const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
        if (canvas) { syncCanvasSize(canvas); renderCanvas(canvas); }
      });
    }
  }, [currentTool, fullscreen, renderCanvas, syncCanvasSize]);

  // --- Keyboard ---
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Undo (Ctrl+Z)
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        const si = d.sliceIndex;
        const stack = undoStackRef.current[si] || [];
        const prev = stack.pop();
        if (prev !== undefined) {
          if (!redoStackRef.current[si]) redoStackRef.current[si] = [];
          redoStackRef.current[si].push(JSON.parse(JSON.stringify(annotations)));
          const map = { ...(d.sliceAnnotationsMap || {}) };
          map[si] = prev;
          updateNodeData(id, { annotations: prev, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
          setSelectedIdx(-1);
        }
        return;
      }
      // Redo (Ctrl+Y or Ctrl+Shift+Z)
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        e.stopPropagation();
        const si = d.sliceIndex;
        const stack = redoStackRef.current[si] || [];
        const next = stack.pop();
        if (next !== undefined) {
          if (!undoStackRef.current[si]) undoStackRef.current[si] = [];
          undoStackRef.current[si].push(JSON.parse(JSON.stringify(annotations)));
          const map = { ...(d.sliceAnnotationsMap || {}) };
          map[si] = next;
          updateNodeData(id, { annotations: next, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
          setSelectedIdx(-1);
        }
        return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.stopPropagation(); // Prevent React Flow from deleting the node
        deleteSelected();
        e.preventDefault();
      }
      if (e.key === ' ' || e.key === 'Space') {
        // Close polygon
        const dr = drawingRef.current;
        if (dr.active && dr.tool === 'polygon' && dr.points.length >= 3) {
          const color = ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length];
          commitAnnotation({
            type: 'polygon',
            points: [...dr.points],
            label: `Poly${annotations.length + 1}`,
            color,
          });
          dr.active = false;
          dr.points = [];
          dr.currentMouse = null;
          e.preventDefault();
        }
      }
      if (e.key === 'Escape') {
        e.stopPropagation();
        // Cancel current drawing
        const dr = drawingRef.current;
        dr.active = false;
        dr.points = [];
        dr.startPoint = null;
        dr.currentRadius = 0;
        dr.currentMouse = null;
        setSelectedIdx(-1);
        requestRender();
        if (fullscreen) setFullscreen(false);
      }
    },
    [deleteSelected, annotations, annotations.length, commitAnnotation, fullscreen, d.sliceAnnotationsMap, d.sliceIndex, id, updateNodeData, pushHistory],
  );

  // --- Force re-render (for live drawing preview) ---
  const requestRender = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    animFrameRef.current = requestAnimationFrame(() => {
      const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
      if (canvas) {
        syncCanvasSize(canvas);
        renderCanvas(canvas);
      }
    });
  }, [fullscreen, renderCanvas, syncCanvasSize]);

  // --- Double-click: edit shape label or reset zoom ---
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      // Check if double-clicked on an annotation
      const coords = getCanvasCoords(e);
      if (coords) {
        const imgPt = canvasToImage(coords.cx, coords.cy, viewState);
        const hitIdx = hitTestAnnotation(imgPt);
        if (hitIdx >= 0) {
          // Open label editor for this annotation
          setSelectedIdx(hitIdx);
          setLabelEditor({ open: true, mode: 'edit', pendingShape: null, editIdx: hitIdx });
          return;
        }
      }
      // No shape hit → reset zoom
      const canvas = fullscreen ? canvasFSRef.current : canvasNodeRef.current;
      fitImageToCanvas(canvas);
    },
    [fullscreen, fitImageToCanvas, getCanvasCoords, canvasToImage, viewState, hitTestAnnotation],
  );

  // --- Right-click context menu on annotations ---
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const coords = getCanvasCoords(e);
      if (!coords) return;
      const imgPt = canvasToImage(coords.cx, coords.cy, viewState);
      const hitIdx = hitTestAnnotation(imgPt);
      if (hitIdx >= 0) {
        setSelectedIdx(hitIdx);
        setContextMenu({ x: e.clientX, y: e.clientY, annIdx: hitIdx });
      }
    },
    [getCanvasCoords, canvasToImage, viewState, hitTestAnnotation],
  );

  // --- Navigate from PositionNavigator ---
  const handleNavigate = useCallback(
    (newOffsetX: number, newOffsetY: number) => {
      setViewState((prev) => ({ ...prev, offsetX: newOffsetX, offsetY: newOffsetY }));
    },
    [],
  );

  // --- Cursor ---
  const cursorForTool = (): string => {
    switch (currentTool) {
      case 'rect': return 'crosshair';
      case 'pan': return 'grab';
      case 'point': return 'crosshair';
      case 'polygon': return 'crosshair';
      case 'circle': return 'crosshair';
      case 'freehand': return 'crosshair';
      case 'eraser': return 'none'; // custom circle cursor drawn on canvas
      default: return 'default';
    }
  };

  // --- Undo / Redo callbacks (for toolbar buttons) ---
  const handleUndo = useCallback(() => {
    const si = d.sliceIndex;
    const stack = undoStackRef.current[si] || [];
    const prev = stack.pop();
    if (prev !== undefined) {
      if (!redoStackRef.current[si]) redoStackRef.current[si] = [];
      redoStackRef.current[si].push(JSON.parse(JSON.stringify(annotations)));
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[si] = prev;
      updateNodeData(id, { annotations: prev, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
      setSelectedIdx(-1);
    }
  }, [d.sliceIndex, d.sliceAnnotationsMap, annotations, id, updateNodeData]);

  const handleRedo = useCallback(() => {
    const si = d.sliceIndex;
    const stack = redoStackRef.current[si] || [];
    const next = stack.pop();
    if (next !== undefined) {
      if (!undoStackRef.current[si]) undoStackRef.current[si] = [];
      undoStackRef.current[si].push(JSON.parse(JSON.stringify(annotations)));
      const map = { ...(d.sliceAnnotationsMap || {}) };
      map[si] = next;
      updateNodeData(id, { annotations: next, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
      setSelectedIdx(-1);
    }
  }, [d.sliceIndex, d.sliceAnnotationsMap, annotations, id, updateNodeData]);

  // --- Auto-detected view badge ---
  const autoDetectedBadge = d.metadata
    ? (d.metadata.ImageOrientationPatient || d.metadata.affine ? '(auto)' : '')
    : '';

  // --- Toolbar renderer (shared between inline and fullscreen) ---
  const renderToolbar = (compact = false) => (
    <div style={{ ...toolbarStyle, gap: compact ? 4 : 3 }}>
      {([
        ['rect', IconRect, 'Rect'],
        ['polygon', IconPolygon, 'Poly'],
        ['circle', IconCircle, 'Circle'],
        ['freehand', IconFreehand, 'Free'],
        ['point', IconPoint, 'Point'],
        ['pan', IconPan, 'Pan'],
      ] as [AnnotationTool, React.ReactNode, string][]).map(([tool, icon, label]) => (
        <button
          key={tool}
          onClick={() => setTool(tool)}
          style={currentTool === tool ? activeTool : toolBtnBase}
          title={label}
        >
          <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {icon}
            <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 40, lineHeight: 1 }}>{label}</span>
          </span>
        </button>
      ))}
      {/* Eraser tool + size picker arrow */}
      <div style={{ position: 'relative', display: 'flex', alignItems: 'stretch' }}>
        <button
          onClick={() => setTool('eraser')}
          style={{ ...(currentTool === 'eraser' ? activeTool : toolBtnBase), borderRadius: '4px 0 0 4px', borderRight: 'none' }}
          title="Eraser"
        >
          <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {IconEraser}
            <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', lineHeight: 1 }}>Erase</span>
          </span>
        </button>
        <button
          onClick={() => setShowEraserPicker((v) => !v)}
          style={{ ...(currentTool === 'eraser' ? activeTool : toolBtnBase), padding: '4px 3px', borderRadius: '0 4px 4px 0', minWidth: 14, width: 14 }}
          title="Eraser size"
        >
          <svg width="8" height="8" viewBox="0 0 8 8" fill="currentColor"><polygon points="0,0 8,0 4,7" /></svg>
        </button>
        {showEraserPicker && (
          <div
            style={{
              position: 'absolute', top: '100%', left: 0, zIndex: 9999,
              background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
              borderRadius: 6, padding: '10px 14px', marginTop: 4,
              boxShadow: '0 4px 16px rgba(0,0,0,0.4)', minWidth: 190,
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
              Eraser size: <strong>{eraserRadius}px</strong>
            </div>
            <input
              type="range" min={4} max={120} step={2}
              value={eraserRadius}
              onChange={(e) => setEraserRadius(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-green)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>
              <span>4px</span><span>120px</span>
            </div>
            <button
              onClick={() => setShowEraserPicker(false)}
              style={{ ...toolBtnBase, marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              Close
            </button>
          </div>
        )}
      </div>
      <span style={{ flex: 1 }} />
      {/* Undo / Redo */}
      <button
        onClick={handleUndo}
        style={{ ...toolBtnBase, opacity: (undoStackRef.current[d.sliceIndex]?.length ?? 0) > 0 ? 1 : 0.35 }}
        title="Undo (Ctrl+Z)"
        disabled={(undoStackRef.current[d.sliceIndex]?.length ?? 0) === 0}
      >
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          {IconUndo}
          <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', lineHeight: 1 }}>Undo</span>
        </span>
      </button>
      <button
        onClick={handleRedo}
        style={{ ...toolBtnBase, opacity: (redoStackRef.current[d.sliceIndex]?.length ?? 0) > 0 ? 1 : 0.35 }}
        title="Redo (Ctrl+Y)"
        disabled={(redoStackRef.current[d.sliceIndex]?.length ?? 0) === 0}
      >
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          {IconRedo}
          <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', lineHeight: 1 }}>Redo</span>
        </span>
      </button>
      {annotations.length > 0 && (
        <button onClick={clearAll} style={toolBtnBase} title="Clear all annotations">
          <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {IconClear}
            <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', lineHeight: 1 }}>Clear</span>
          </span>
        </button>
      )}
      {d.imageBase64 && (
        <button onClick={downloadSlice} style={toolBtnBase} title="Download slice with annotations (PNG)">
          <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {IconDownload}
            <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 0.2, textTransform: 'uppercase', lineHeight: 1 }}>Save</span>
          </span>
        </button>
      )}
    </div>
  );

  // --- Zoom controls renderer ---
  const renderZoomControls = () => (
    <div style={controlsRowStyle}>
      <button onClick={zoomOut} style={smallBtnStyle} title="Zoom out">−</button>
      <button onClick={zoomReset} style={smallBtnStyle} title="Reset zoom">⊙</button>
      <button onClick={zoomIn} style={smallBtnStyle} title="Zoom in">+</button>
      <span style={badgeStyle}>
        🔍 {Math.round(viewState.scale * 100)}%
      </span>
      <span style={{ flex: 1 }} />
      <button
        onClick={() => {
          updateNodeData(id, { showLabels: !showLabels } as Partial<InteractiveAnnotatorNodeData>);
        }}
        style={{
          ...smallBtnStyle,
          background: showLabels ? 'var(--accent-green)' : undefined,
          color: showLabels ? '#fff' : undefined,
        }}
        title={showLabels ? 'Hide labels' : 'Show labels'}
      >
        {IconLabels}
      </button>
      <button
        onClick={() => setShowNavigator((v) => !v)}
        style={{
          ...smallBtnStyle,
          background: showNavigator ? 'var(--accent-green)' : undefined,
          color: showNavigator ? '#fff' : undefined,
        }}
        title={showNavigator ? 'Hide navigator' : 'Show navigator'}
      >
        {IconNavigator}
      </button>
      <button
        onClick={() => setFullscreen(!fullscreen)}
        style={smallBtnStyle}
        title={fullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
      >
        {fullscreen ? '⊗' : '⛶'}
      </button>
    </div>
  );

  // --- Canvas element renderer ---
  const renderCanvasEl = (ref: React.RefObject<HTMLCanvasElement | null>, areaStyle?: React.CSSProperties) => (
    <div
      style={{ ...canvasWrapperStyle, ...areaStyle, cursor: cursorForTool() }}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onKeyDown={handleKeyDown}
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
      tabIndex={0}
    >
      <canvas ref={ref} style={canvasStyle} />
      {showNavigator && imgDimensions && (
        <PositionNavigator
          imageObj={imageObjRef.current}
          imgDimensions={imgDimensions}
          viewState={viewState}
          canvasRef={ref}
          onNavigate={handleNavigate}
        />
      )}
      {loading && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.5)',
            color: '#fff',
            fontSize: 13,
          }}
        >
          Loading…
        </div>
      )}
      {labelEditor.open && (
        <LabelEditorPopup
          initialLabel={
            labelEditor.mode === 'edit' && labelEditor.editIdx >= 0
              ? annotations[labelEditor.editIdx]?.label || ''
              : labelEditor.pendingShape?.label || ''
          }
          initialColor={
            labelEditor.mode === 'edit' && labelEditor.editIdx >= 0
              ? annotations[labelEditor.editIdx]?.color || ANNOTATION_COLORS[0]
              : labelEditor.pendingShape?.color || ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length]
          }
          onConfirm={handleLabelConfirm}
          onCancel={handleLabelCancel}
        />
      )}
    </div>
  );

  // --- Fullscreen portal ---
  const fullscreenPortal = fullscreen
    ? createPortal(
        <div style={fullscreenOverlayStyle} onKeyDown={handleKeyDown} tabIndex={0}>
          {/* Header */}
          <div style={fullscreenHeaderStyle}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 14 }}>
              ✏️ Interactive Annotator — Fullscreen
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
              {renderToolbar(true)}
            </div>
            {renderZoomControls()}
          </div>
          {/* Canvas */}
          {renderCanvasEl(canvasFSRef, {
            ...fullscreenCanvasArea,
            aspectRatio: 'unset',
            borderRadius: 0,
            border: 'none',
          })}
          {/* Footer with navigation */}
          {hasSession && (
            <div
              style={{
                padding: '8px 16px',
                background: 'rgba(26, 27, 46, 0.95)',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                flexShrink: 0,
              }}
            >
              <button onClick={goPrev} style={navBtnStyle} disabled={d.sliceIndex <= 0}>◀</button>
              <input
                type="range"
                min={0}
                max={Math.max(0, d.totalSlices - 1)}
                value={d.sliceIndex}
                onChange={handleSlider}
                style={sliderStyle}
              />
              <button onClick={goNext} style={navBtnStyle} disabled={d.sliceIndex >= d.totalSlices - 1}>▶</button>
              <span style={sliceInfoStyle}>
                {d.sliceIndex + 1} / {d.totalSlices}
              </span>
              <span style={badgeStyle}>
                {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}
        </div>,
        document.body,
      )
    : null;

  // --- Inline render ---
  return (
    <>
      <BaseNode
        nodeId={id}
        nodeType="interactiveAnnotator"
        title="Interactive Annotator"
        icon="✏️"
        color="var(--accent-green)"
        status={d.status}
        error={d.error}
        hasInput={true}
        hasOutput={true}
        info={ANNOTATOR_INFO}
      >
        <div style={viewerContainerStyle}>
          {/* View selector */}
          {hasSession && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ ...labelStyle, marginBottom: 0 }}>
                  View Plane{' '}
                  {autoDetectedBadge && (
                    <span style={{ fontSize: 9, color: 'var(--accent-green)', fontWeight: 400 }}>
                      {autoDetectedBadge}
                    </span>
                  )}
                </label>
              </div>
              <div style={viewSelectorStyle}>
                {VIEWS.map((v) => (
                  <button
                    key={v}
                    onClick={() => handleViewChange(v)}
                    style={{
                      ...viewBtnBase,
                      background: currentView === v ? 'var(--accent-green)' : 'var(--bg-tertiary)',
                      color: currentView === v ? '#fff' : 'var(--text-secondary)',
                      borderColor: currentView === v ? 'var(--accent-green)' : 'var(--border-color)',
                    }}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </>
          )}

          {/* Canvas */}
          {renderCanvasEl(canvasNodeRef)}

          {/* Tool selector (below canvas) */}
          {hasSession && d.imageBase64 && renderToolbar()}

          {/* AI Segmentation panel - runs SAM2 on the currently visible slice */}
          {hasSession && d.imageBase64 && (
            <div style={{
              marginTop: 6,
              padding: '6px 8px',
              background: 'var(--bg-tertiary)',
              borderRadius: 6,
              border: '1px solid var(--accent-purple)30',
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 8,
                marginBottom: 6,
              }}>
                <div>
                  <div style={{
                    color: 'var(--accent-purple)',
                    fontSize: 10,
                    fontWeight: 800,
                    letterSpacing: 0.6,
                    textTransform: 'uppercase',
                  }}>
                    AI Segmentation
                  </div>
                  <div style={{
                    color: 'var(--text-muted)',
                    fontSize: 10,
                    lineHeight: 1.35,
                    marginTop: 2,
                  }}>
                    Current {d.view || 'axial'} slice {d.sliceIndex}
                  </div>
                </div>
                <span style={{
                  color: 'var(--accent-purple)',
                  fontSize: 10,
                  fontWeight: 800,
                  whiteSpace: 'nowrap',
                }}>
                  {(connectedAutoSeg?.configName || 'fast').replace(/_/g, ' ')}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <button
                  onClick={handleRunAutoSegmentation}
                  disabled={segRunning}
                  style={{
                    flex: 1,
                    padding: '7px 12px',
                    borderRadius: 6,
                    border: 'none',
                    background: segRunning
                      ? 'var(--text-muted)'
                      : 'linear-gradient(135deg, var(--accent-purple), #7c4dff)',
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: segRunning ? 'wait' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    transition: 'opacity 0.15s',
                    opacity: segRunning ? 0.7 : 1,
                  }}
                >
                  {segRunning ? (
                    <>Running SAM2...</>
                  ) : (
                    <>Run SAM2 on Current Slice</>
                  )}
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                <button
                  onClick={handleRunPromptSegmentation}
                  disabled={segRunning || !activePrompt}
                  style={{
                    flex: 1,
                    padding: '7px 12px',
                    borderRadius: 6,
                    border: '1px solid var(--accent-orange)',
                    background: activePrompt && !segRunning
                      ? 'rgba(255, 152, 0, 0.14)'
                      : 'var(--bg-secondary)',
                    color: activePrompt ? 'var(--accent-orange)' : 'var(--text-muted)',
                    fontSize: 12,
                    fontWeight: 700,
                    cursor: segRunning ? 'wait' : activePrompt ? 'pointer' : 'not-allowed',
                  }}
                >
                  Run Prompt SAM2
                </button>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 78, textAlign: 'right' }}>
                  {activePrompt
                    ? `${activePrompt.points.length} point / ${activePrompt.boxes.length} box`
                    : 'No prompt'}
                </span>
              </div>
              {segMessage && (
                <div style={{
                  marginTop: 4,
                  fontSize: 10,
                  color: segMessage.startsWith('Done:') ? 'var(--accent-green)' : 'var(--accent-red)',
                }}>
                  {segMessage}
                </div>
              )}
            </div>
          )}

          {hasSession && d.imageBase64 && d.labelSuggestions && d.labelSuggestions.length > 0 && (
            <div style={{
              marginTop: 6,
              padding: '7px 8px',
              background: 'rgba(76, 175, 139, 0.08)',
              borderRadius: 6,
              border: '1px solid rgba(76, 175, 139, 0.25)',
            }}>
              <div style={{
                color: 'var(--accent-green)',
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: 0.6,
                textTransform: 'uppercase',
                marginBottom: 6,
              }}>
                Suggested Labels
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {d.labelSuggestions.map((label) => (
                  <span
                    key={label}
                    style={{
                      padding: '3px 7px',
                      borderRadius: 4,
                      border: '1px solid rgba(76, 175, 139, 0.34)',
                      color: 'var(--accent-green)',
                      background: 'rgba(76, 175, 139, 0.1)',
                      fontSize: 10,
                      fontWeight: 800,
                      lineHeight: 1.2,
                    }}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Zoom controls */}
          {hasSession && d.imageBase64 && renderZoomControls()}

          {/* Placeholder */}
          {!hasSession && (
            <NodeHint style={{ textAlign: 'left' }}>
              Connect a Data Loader to start annotating.
            </NodeHint>
          )}

          {/* Slice navigation */}
          {hasSession && d.totalSlices > 0 && (
            <div style={controlsRowStyle}>
              <button onClick={goPrev} style={navBtnStyle} disabled={d.sliceIndex <= 0}>
                ◀
              </button>
              <input
                type="range"
                min={0}
                max={Math.max(0, d.totalSlices - 1)}
                value={d.sliceIndex}
                onChange={handleSlider}
                style={sliderStyle}
              />
              <button onClick={goNext} style={navBtnStyle} disabled={d.sliceIndex >= d.totalSlices - 1}>
                ▶
              </button>
              <span style={sliceInfoStyle}>
                {d.sliceIndex + 1} / {d.totalSlices}
              </span>
            </div>
          )}

          {/* Annotation count & metadata */}
          {hasSession && (
            <div style={metaRowStyle}>
              <span>{annotations.length} annotation{annotations.length !== 1 ? 's' : ''}</span>
              {d.volumeShape && (
                <span>
                  vol {d.volumeShape.join(' × ')}
                </span>
              )}
            </div>
          )}
        </div>
      </BaseNode>

      {/* Fullscreen portal */}
      {fullscreenPortal}

      {/* Right-click context menu portal */}
      {contextMenu && createPortal(
        <>
          {/* Backdrop — click outside to close */}
          <div
            style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
            onClick={() => setContextMenu(null)}
            onContextMenu={(e) => { e.preventDefault(); setContextMenu(null); }}
          />
          <div
            style={{
              position: 'fixed',
              left: contextMenu.x,
              top: contextMenu.y,
              zIndex: 9999,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: '4px 0',
              boxShadow: '0 4px 16px rgba(0,0,0,0.45)',
              minWidth: 148,
            }}
          >
            <button
              style={contextMenuItemStyle}
              onClick={() => {
                setLabelEditor({ open: true, mode: 'edit', pendingShape: null, editIdx: contextMenu.annIdx });
                setContextMenu(null);
              }}
            >
              ✏️ Rename
            </button>
            <button
              style={{ ...contextMenuItemStyle, color: 'var(--accent-red, #e57373)' }}
              onClick={() => {
                pushHistory(annotations);
                const newAnns = annotations.filter((_, i) => i !== contextMenu.annIdx);
                const map = { ...(d.sliceAnnotationsMap || {}) };
                map[d.sliceIndex] = newAnns;
                updateNodeData(id, { annotations: newAnns, sliceAnnotationsMap: map } as Partial<InteractiveAnnotatorNodeData>);
                setSelectedIdx(-1);
                setContextMenu(null);
              }}
            >
              🗑️ Delete
            </button>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}

export default memo(InteractiveAnnotatorNode);
