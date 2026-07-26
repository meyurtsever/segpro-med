/**
 * SliceViewerNode
 * ================
 * Inline medical image slice viewer with navigation controls.
 * ComfyUI-style preview node with prev/next buttons, slider, and view selector.
 *
 * Features:
 *  - Auto-selects view plane from upstream metadata (IOP / affine)
 *  - Zoom in/out via mouse wheel, buttons, or pinch
 *  - Coordinate selection via click — stores (x, y, slice, view) for AI pipelines
 *
 * WF-2 Step: Slice Viewer — displays slices from loaded sessions.
 * Supports all formats from DataLoader or FormatConverter output.
 */

import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { SliceViewerNodeData, SliceCoordinate } from '../types/nodes';
import * as api from '../api/client';
import type { PatientSearchResult } from '../api/client';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';

// ---------------------------------------------------------------------------
// Info modal content
// ---------------------------------------------------------------------------

const SLICE_VIEWER_INFO: NodeInfo = {
  description:
    'Inline slice viewer for medical imaging volumes. Supports direct patient search by ICD-10 code or clinical keyword — no separate Data Loader required. When a patient with a segmentation file is loaded, an overlay toggle appears.',
  inputs: [
    'Session ID from a Data Loader node (direct connection)',
    'Output file path from a Format Converter node (auto-loads on execution)',
    'Or: search by ICD-10 code / keyword at the top of the node — standalone use',
  ],
  outputs: [],
  tips: [
    'Type an ICD-10 code (e.g. "D18.02") or condition name (e.g. "glioblastoma") to instantly load a patient',
    'The 🩻 icon in search results means a segmentation file is available for that patient',
    'Toggle the segmentation overlay with the On/Off button once a patient is loaded',
    'Use the slider or ◀ ▶ buttons to navigate through slices',
    'Switch between Axial, Coronal, and Sagittal views — auto-detected from metadata',
    'Scroll wheel on the image to zoom in/out, or use the −/+ buttons',
    'Click on the image to select a coordinate — stored for AI pipelines',
  ],
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const viewerContainerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};

const imageWrapperStyle: React.CSSProperties = {
  position: 'relative',
  width: '100%',
  aspectRatio: '1',
  background: '#000',
  borderRadius: 6,
  overflow: 'hidden',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  border: '1px solid var(--border-color)',
  cursor: 'crosshair',
};

const imageStyle: React.CSSProperties = {
  maxWidth: '100%',
  maxHeight: '100%',
  objectFit: 'contain',
  imageRendering: 'pixelated',
  transformOrigin: 'center center',
  transition: 'transform 0.1s ease-out',
  userSelect: 'none',
  pointerEvents: 'none',
};

const placeholderStyle: React.CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 12,
  textAlign: 'center',
  padding: 20,
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
  accentColor: 'var(--accent-purple)',
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

const coordBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '3px 8px',
  borderRadius: 4,
  fontSize: 10,
  fontFamily: 'monospace',
  background: 'rgba(155, 109, 215, 0.08)',
  border: '1px solid rgba(155, 109, 215, 0.25)',
  color: 'var(--accent-purple)',
};

const coordRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 6,
};

const VIEWS = ['axial', 'coronal', 'sagittal'] as const;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;
const CROSSHAIR_SIZE = 16;

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function SliceViewerNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as SliceViewerNodeData;

  const [loading, setLoading] = useState(false);
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);
  const fetchRef = useRef(0);
  const imgRef = useRef<HTMLImageElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Patient search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<PatientSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loadingPatient, setLoadingPatient] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const searchWrapperRef = useRef<HTMLDivElement>(null);

  const [fullscreen, setFullscreen] = useState(false);

  const zoom = d.zoom ?? 1;
  const currentView = d.view || 'axial';
  const hasSession = Boolean(d.sessionId);
  const showLabels = d.showLabels !== false;

  // Derived: seg path to pass to the backend (only when overlay is on)
  const activeSegPath = (d.showOverlay && d.segPath) ? d.segPath : undefined;

  // --- Fetch a slice from the API ---
  const fetchSlice = useCallback(
    async (
      sessionId: string,
      sliceIdx: number,
      view: string,
      segPath?: string,
      overlayLabels: boolean = true,
    ) => {
      const fetchId = ++fetchRef.current;
      setLoading(true);
      try {
        const res = await api.getSlice(sessionId, sliceIdx, view, segPath, undefined, Boolean(segPath && overlayLabels));
        if (fetchRef.current !== fetchId) return;
        updateNodeData(id, {
          status: 'success',
          error: undefined,
          imageBase64: res.image_base64,
          sliceIndex: res.slice_index,
          totalSlices: res.total_slices,
        });
        setDimensions({ w: res.width, h: res.height });
      } catch (err) {
        if (fetchRef.current !== fetchId) return;
        updateNodeData(id, {
          status: 'error',
          error: err instanceof Error ? err.message : 'Failed to fetch slice',
        });
      } finally {
        if (fetchRef.current === fetchId) setLoading(false);
      }
    },
    [id, updateNodeData],
  );

  // --- Auto-fetch when sessionId becomes available ---
  useEffect(() => {
    if (d.sessionId && !d.imageBase64) {
      const canonicalView =
        ((d.metadata?.canonical_view as string) as 'axial' | 'coronal' | 'sagittal') ||
        d.view ||
        'axial';
      if (canonicalView !== (d.view || 'axial')) {
        updateNodeData(id, { view: canonicalView });
      }
      fetchSlice(d.sessionId, d.sliceIndex || 0, canonicalView, activeSegPath, showLabels);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d.sessionId]);

  // --- Navigation ---
  const goPrev = useCallback(() => {
    if (!d.sessionId || d.sliceIndex <= 0) return;
    const next = d.sliceIndex - 1;
    updateNodeData(id, { sliceIndex: next });
    fetchSlice(d.sessionId, next, d.view || 'axial', activeSegPath, showLabels);
  }, [d.sessionId, d.sliceIndex, d.view, id, updateNodeData, fetchSlice, activeSegPath, showLabels]);

  const goNext = useCallback(() => {
    if (!d.sessionId || d.sliceIndex >= d.totalSlices - 1) return;
    const next = d.sliceIndex + 1;
    updateNodeData(id, { sliceIndex: next });
    fetchSlice(d.sessionId, next, d.view || 'axial', activeSegPath, showLabels);
  }, [d.sessionId, d.sliceIndex, d.totalSlices, d.view, id, updateNodeData, fetchSlice, activeSegPath, showLabels]);

  const handleSlider = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!d.sessionId) return;
      const idx = parseInt(e.target.value, 10);
      updateNodeData(id, { sliceIndex: idx });
      fetchSlice(d.sessionId, idx, d.view || 'axial', activeSegPath, showLabels);
    },
    [d.sessionId, d.view, id, updateNodeData, fetchSlice, activeSegPath, showLabels],
  );

  const handleViewChange = useCallback(
    (view: 'axial' | 'coronal' | 'sagittal') => {
      if (!d.sessionId) return;
      updateNodeData(id, { view, sliceIndex: 0, selectedCoord: undefined });
      fetchSlice(d.sessionId, 0, view, activeSegPath, showLabels);
    },
    [d.sessionId, id, updateNodeData, fetchSlice, activeSegPath, showLabels],
  );

  // --- Zoom ---
  const setZoom = useCallback(
    (newZoom: number) => {
      const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
      updateNodeData(id, { zoom: Math.round(clamped * 100) / 100 });
    },
    [id, updateNodeData],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
      setZoom(zoom + delta);
    },
    [zoom, setZoom],
  );

  const zoomIn = useCallback(() => setZoom(zoom + ZOOM_STEP), [zoom, setZoom]);
  const zoomOut = useCallback(() => setZoom(zoom - ZOOM_STEP), [zoom, setZoom]);
  const zoomReset = useCallback(() => setZoom(1), [setZoom]);

  // --- Debounced patient search ---
  useEffect(() => {
    if (searchQuery.trim().length < 2) {
      setSearchResults([]);
      setShowDropdown(false);
      setSearchError(null);
      return;
    }
    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await api.searchPatients(searchQuery.trim());
        setSearchResults(res.results);
        setShowDropdown(true);
        setSearchError(null);
      } catch (err) {
        setSearchResults([]);
        setSearchError(err instanceof Error ? err.message : 'Patient search failed');
      } finally {
        setIsSearching(false);
      }
    }, 350);
    return () => {
      clearTimeout(timer);
      setIsSearching(false);
    };
  }, [searchQuery]);

  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (searchWrapperRef.current && !searchWrapperRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  // --- Escape key exits fullscreen ---
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setFullscreen(false); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [fullscreen]);

  const handlePatientSelect = useCallback(
    async (result: PatientSearchResult) => {
      setShowDropdown(false);
      setSearchQuery('');
      if (!result.path) return;
      setLoadingPatient(true);
      updateNodeData(id, { status: 'running', error: undefined });
      try {
        const loaded = await api.loadDataFromPath(result.path);
        const canonicalView =
          ((loaded.metadata as Record<string, unknown>)?.canonical_view as
            | 'axial'
            | 'coronal'
            | 'sagittal') ||
          d.view ||
          'axial';
        updateNodeData(id, {
          status: 'success',
          error: undefined,
          sessionId: loaded.session_id,
          segPath: result.segmentation_path || undefined,
          showOverlay: false,
          volumeShape: loaded.volume_shape,
          metadata: loaded.metadata as Record<string, unknown>,
          sliceIndex: 0,
          view: canonicalView,
          imageBase64: undefined,
          showLabels: d.showLabels ?? true,
        });
        fetchSlice(loaded.session_id, 0, canonicalView);
      } catch (err) {
        updateNodeData(id, {
          status: 'error',
          error: err instanceof Error ? err.message : 'Failed to load patient',
        });
      } finally {
        setLoadingPatient(false);
      }
    },
    [id, updateNodeData, fetchSlice, d.view, d.showLabels],
  );

  const toggleOverlay = useCallback(() => {
    const newShow = !d.showOverlay;
    updateNodeData(id, { showOverlay: newShow });
    if (d.sessionId) {
      fetchSlice(
        d.sessionId,
        d.sliceIndex || 0,
        d.view || 'axial',
        newShow ? d.segPath : undefined,
        showLabels,
      );
    }
  }, [d.sessionId, d.showOverlay, d.segPath, d.sliceIndex, d.view, id, updateNodeData, fetchSlice, showLabels]);

  const toggleLabels = useCallback(() => {
    const nextShowLabels = !showLabels;
    updateNodeData(id, { showLabels: nextShowLabels });
    if (d.sessionId && d.showOverlay && d.segPath) {
      fetchSlice(
        d.sessionId,
        d.sliceIndex || 0,
        d.view || 'axial',
        d.segPath,
        nextShowLabels,
      );
    }
  }, [d.sessionId, d.showOverlay, d.segPath, d.sliceIndex, d.view, id, updateNodeData, fetchSlice, showLabels]);

  // --- Coordinate selection ---
  const handleImageClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!d.imageBase64 || !dimensions || !imgRef.current) return;

      const imgEl = imgRef.current;
      const imgRect = imgEl.getBoundingClientRect();

      // Click position relative to the rendered image
      const relX = e.clientX - imgRect.left;
      const relY = e.clientY - imgRect.top;

      // Convert to image pixel coordinates (accounting for zoom)
      const imgPixelX = Math.round((relX / imgRect.width) * dimensions.w);
      const imgPixelY = Math.round((relY / imgRect.height) * dimensions.h);

      // Clamp to valid range
      const x = Math.max(0, Math.min(dimensions.w - 1, imgPixelX));
      const y = Math.max(0, Math.min(dimensions.h - 1, imgPixelY));

      const coord: SliceCoordinate = {
        x,
        y,
        sliceIndex: d.sliceIndex,
        view: currentView,
      };

      updateNodeData(id, { selectedCoord: coord });
    },
    [d.imageBase64, d.sliceIndex, dimensions, currentView, id, updateNodeData],
  );

  const clearCoord = useCallback(() => {
    updateNodeData(id, { selectedCoord: undefined });
  }, [id, updateNodeData]);

  // --- Crosshair position in CSS % ---
  const crosshairPos = (() => {
    if (!d.selectedCoord || !dimensions) return null;
    if (d.selectedCoord.view !== currentView) return null;
    return {
      leftPct: (d.selectedCoord.x / dimensions.w) * 100,
      topPct: (d.selectedCoord.y / dimensions.h) * 100,
    };
  })();

  // --- Auto-detected view badge ---
  const autoDetectedBadge = d.metadata
    ? (d.metadata.ImageOrientationPatient || d.metadata.affine ? '(auto)' : '')
    : '';

  return (
    <>
    <BaseNode
      nodeId={id}
      nodeType="sliceViewer"
      title="Slice Viewer"
      icon="🖼️"
      color="var(--accent-purple)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={false}
      info={SLICE_VIEWER_INFO}
      minWidth={380}
      maxWidth={420}
    >
      {/* ── Patient Search ──────────────────────────────────────────────── */}
      <div ref={searchWrapperRef} style={{ position: 'relative', marginBottom: 8 }}>
        <div style={{ position: 'relative' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="🔍 ICD-10 or keyword — e.g. 'D18.02', 'glioblastoma'…"
            style={{
              width: '100%',
              padding: '6px 28px 6px 8px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: 5,
              color: 'var(--text-primary)',
              fontSize: 11,
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily: 'inherit',
            }}
            onFocus={() => searchResults.length > 0 && setShowDropdown(true)}
          />
          {(isSearching || loadingPatient) && (
            <span style={{
              position: 'absolute', right: 8, top: '50%',
              transform: 'translateY(-50%)', fontSize: 11,
              color: 'var(--text-muted)', pointerEvents: 'none',
            }}>⏳</span>
          )}
        </div>

        {showDropdown && (
          <div style={{
            position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 200,
            background: 'var(--bg-secondary, #1e1e2e)',
            border: '1px solid var(--border-color)',
            borderRadius: 6, marginTop: 2, maxHeight: 200, overflowY: 'auto',
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
          }}>
            {searchResults.length === 0 ? (
              <div style={{ padding: '8px 10px', fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                No patients found.
              </div>
            ) : (
              searchResults.map((r) => (
                <button
                  key={r.display_name}
                  onClick={() => handlePatientSelect(r)}
                  style={{
                    display: 'block', width: '100%', background: 'transparent',
                    border: 'none', borderBottom: '1px solid var(--border-color)',
                    cursor: 'pointer', padding: '7px 10px',
                    textAlign: 'left', color: 'var(--text-primary)',
                  }}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.background = 'var(--bg-tertiary)')}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.background = 'transparent')}
                >
                  <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 2 }}>{r.display_name}</div>
                  <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                    {r.icd10_code && (
                      <span style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 10,
                        fontSize: 10, fontWeight: 600,
                        background: 'rgba(89,152,222,0.15)', color: 'var(--accent-blue)',
                        border: '1px solid rgba(89,152,222,0.3)',
                      }}>{r.icd10_code}</span>
                    )}
                    {r.icd10_description && (
                      <span style={{ fontSize: 10, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>
                        {r.icd10_description}
                      </span>
                    )}
                    {r.segmentation_path && (
                      <span style={{
                        display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
                        background: 'var(--accent-green)', flexShrink: 0,
                      }} title="Annotation available" />
                    )}
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {searchError && (
        <div style={{ marginBottom: 8, color: 'var(--accent-red)', fontSize: 10 }}>
          {searchError}
        </div>
      )}

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
                    background:
                      currentView === v ? 'var(--accent-purple)' : 'var(--bg-tertiary)',
                    color: currentView === v ? '#fff' : 'var(--text-secondary)',
                    borderColor:
                      currentView === v ? 'var(--accent-purple)' : 'var(--border-color)',
                  }}
                >
                  {v}
                </button>
              ))}
            </div>
          </>
        )}

        {/* Image display with zoom and coordinate selection */}
        <div
          ref={wrapperRef}
          style={imageWrapperStyle}
          onWheel={handleWheel}
          onClick={handleImageClick}
        >
          {loading && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(0,0,0,0.6)',
                zIndex: 3,
                color: 'var(--accent-purple)',
                fontSize: 22,
              }}
            >
              ⏳
            </div>
          )}
          {d.imageBase64 ? (
            <>
              <img
                ref={imgRef}
                src={`data:image/png;base64,${d.imageBase64}`}
                alt={`Slice ${d.sliceIndex} — ${currentView}`}
                style={{
                  ...imageStyle,
                  transform: `scale(${zoom})`,
                }}
                draggable={false}
              />
              {/* Crosshair overlay */}
              {crosshairPos && (
                <svg
                  style={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    zIndex: 2,
                  }}
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  <line
                    x1={crosshairPos.leftPct}
                    y1={Math.max(crosshairPos.topPct - CROSSHAIR_SIZE, 0)}
                    x2={crosshairPos.leftPct}
                    y2={Math.min(crosshairPos.topPct + CROSSHAIR_SIZE, 100)}
                    stroke="var(--accent-green)"
                    strokeWidth="0.4"
                    strokeDasharray="1,0.5"
                  />
                  <line
                    x1={Math.max(crosshairPos.leftPct - CROSSHAIR_SIZE, 0)}
                    y1={crosshairPos.topPct}
                    x2={Math.min(crosshairPos.leftPct + CROSSHAIR_SIZE, 100)}
                    y2={crosshairPos.topPct}
                    stroke="var(--accent-green)"
                    strokeWidth="0.4"
                    strokeDasharray="1,0.5"
                  />
                  <circle
                    cx={crosshairPos.leftPct}
                    cy={crosshairPos.topPct}
                    r="0.8"
                    fill="var(--accent-green)"
                    stroke="#000"
                    strokeWidth="0.2"
                  />
                </svg>
              )}
            </>
          ) : (
            <div style={placeholderStyle}>
              {hasSession
                ? 'Loading slice…'
                : <NodeHint style={{ marginTop: 0, textAlign: 'left' }}>Connect a Data Loader or Format Converter to view slices.</NodeHint>}
            </div>
          )}

          {/* Zoom badge (top-right of image area) */}
          {hasSession && zoom !== 1 && (
            <div
              style={{
                position: 'absolute',
                top: 4,
                right: 4,
                background: 'rgba(0,0,0,0.7)',
                color: '#fff',
                padding: '2px 6px',
                borderRadius: 3,
                fontSize: 10,
                fontFamily: 'monospace',
                zIndex: 2,
              }}
            >
              {Math.round(zoom * 100)}%
            </div>
          )}
        </div>

        {/* Navigation + zoom controls */}
        {hasSession && (
          <>
            {/* Slider row */}
            <div style={controlsRowStyle}>
              <button
                onClick={goPrev}
                disabled={!hasSession || d.sliceIndex <= 0}
                style={{
                  ...navBtnStyle,
                  opacity: d.sliceIndex <= 0 ? 0.3 : 1,
                }}
                title="Previous slice"
              >
                ◀
              </button>

              <input
                type="range"
                min={0}
                max={Math.max((d.totalSlices || 1) - 1, 0)}
                value={d.sliceIndex || 0}
                onChange={handleSlider}
                style={sliderStyle}
              />

              <button
                onClick={goNext}
                disabled={!hasSession || d.sliceIndex >= d.totalSlices - 1}
                style={{
                  ...navBtnStyle,
                  opacity: d.sliceIndex >= d.totalSlices - 1 ? 0.3 : 1,
                }}
                title="Next slice"
              >
                ▶
              </button>
            </div>

            {/* Slice info + zoom buttons */}
            <div style={metaRowStyle}>
              <span style={sliceInfoStyle}>
                Slice {(d.sliceIndex || 0) + 1} / {d.totalSlices || '—'}
              </span>
              <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                <button onClick={zoomOut} style={smallBtnStyle} title="Zoom out">−</button>
                <button
                  onClick={zoomReset}
                  style={{
                    ...smallBtnStyle,
                    fontSize: 9,
                    minWidth: 32,
                    color: zoom === 1 ? 'var(--text-muted)' : 'var(--accent-purple)',
                  }}
                  title="Reset zoom to 100%"
                >
                  {Math.round(zoom * 100)}%
                </button>
                <button onClick={zoomIn} style={smallBtnStyle} title="Zoom in">+</button>
                <button
                  onClick={() => setFullscreen(true)}
                  style={smallBtnStyle}
                  title="Fullscreen (Esc to exit)"
                >
                  ⛶
                </button>
              </div>
              {dimensions && (
                <span>{dimensions.w}×{dimensions.h}</span>
              )}
            </div>

            {/* Segmentation overlay toggle */}
            {d.segPath && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'nowrap' }}>
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  background: d.showOverlay ? 'var(--accent-green)' : 'var(--border-color)',
                  flexShrink: 0,
                }} />
                <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>Annotation</span>
                <button
                  onClick={toggleOverlay}
                  style={{
                    ...smallBtnStyle,
                    minWidth: 42,
                    fontSize: 10,
                    color: d.showOverlay ? 'var(--accent-green)' : 'var(--text-muted)',
                    borderColor: d.showOverlay ? 'var(--accent-green)' : 'var(--border-color)',
                    background: d.showOverlay ? 'rgba(166,227,161,0.1)' : 'var(--bg-tertiary)',
                  }}
                  title={d.showOverlay ? 'Hide annotation overlay' : 'Show annotation overlay'}
                >
                  {d.showOverlay ? 'On' : 'Off'}
                </button>
                <button
                  onClick={toggleLabels}
                  disabled={!d.showOverlay}
                  style={{
                    ...smallBtnStyle,
                    minWidth: 26,
                    fontSize: 12,
                    fontWeight: 800,
                    color: showLabels && d.showOverlay ? '#fff' : 'var(--text-muted)',
                    borderColor: showLabels && d.showOverlay ? 'var(--accent-green)' : 'var(--border-color)',
                    background: showLabels && d.showOverlay ? 'var(--accent-green)' : 'var(--bg-tertiary)',
                    opacity: d.showOverlay ? 1 : 0.45,
                    cursor: d.showOverlay ? 'pointer' : 'not-allowed',
                  }}
                  title={
                    d.showOverlay
                      ? showLabels
                        ? 'Hide segmentation labels'
                        : 'Show segmentation labels'
                      : 'Turn annotations on before showing labels'
                  }
                >
                  T
                </button>
                <span style={{
                  fontSize: 9, color: 'var(--text-muted)', fontFamily: 'monospace',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                }}>
                  {d.segPath.split(/[\\/]/).pop()}
                </span>
              </div>
            )}

            {/* Selected coordinate readout */}
            {d.selectedCoord && (
              <div style={coordRowStyle}>
                <div style={coordBadgeStyle}>
                  <span>📍</span>
                  <span>
                    ({d.selectedCoord.x}, {d.selectedCoord.y})
                  </span>
                  <span style={{ opacity: 0.6 }}>
                    S{d.selectedCoord.sliceIndex + 1} · {d.selectedCoord.view}
                  </span>
                </div>
                <button
                  onClick={clearCoord}
                  style={{
                    ...smallBtnStyle,
                    fontSize: 10,
                    color: 'var(--text-muted)',
                    padding: '2px 5px',
                  }}
                  title="Clear selected coordinate"
                >
                  ✕
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </BaseNode>

    {/* ── Fullscreen portal ─────────────────────────────────────────── */}
    {fullscreen && createPortal(
      <div
        style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: '#0d0e1a',
          display: 'flex', flexDirection: 'column',
        }}
        onWheel={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '8px 16px',
          background: 'rgba(26,27,46,0.95)',
          borderBottom: '1px solid var(--border-color)',
          flexShrink: 0,
          gap: 12,
        }}>
          <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 14 }}>
            🖼️ Slice Viewer — Fullscreen
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* View selector */}
            <div style={{ display: 'flex', gap: 4 }}>
              {VIEWS.map((v) => (
                <button
                  key={v}
                  onClick={() => handleViewChange(v)}
                  style={{
                    ...smallBtnStyle,
                    background: currentView === v ? 'var(--accent-purple)' : 'var(--bg-tertiary)',
                    color: currentView === v ? '#fff' : 'var(--text-muted)',
                    borderColor: currentView === v ? 'var(--accent-purple)' : 'var(--border-color)',
                    textTransform: 'capitalize',
                    fontSize: 11,
                    padding: '3px 10px',
                  }}
                >
                  {v}
                </button>
              ))}
            </div>
            {/* Overlay toggle */}
            {d.segPath && (
              <button
                onClick={toggleOverlay}
                style={{
                  ...smallBtnStyle,
                  fontSize: 11,
                  color: d.showOverlay ? 'var(--accent-green)' : 'var(--text-muted)',
                  borderColor: d.showOverlay ? 'var(--accent-green)' : 'var(--border-color)',
                }}
              >
                {d.showOverlay ? '● Annotation On' : '○ Annotation Off'}
              </button>
            )}
            {d.segPath && (
              <button
                onClick={toggleLabels}
                disabled={!d.showOverlay}
                style={{
                  ...smallBtnStyle,
                  minWidth: 28,
                  fontSize: 12,
                  fontWeight: 800,
                  color: showLabels && d.showOverlay ? '#fff' : 'var(--text-muted)',
                  borderColor: showLabels && d.showOverlay ? 'var(--accent-green)' : 'var(--border-color)',
                  background: showLabels && d.showOverlay ? 'var(--accent-green)' : 'var(--bg-tertiary)',
                  opacity: d.showOverlay ? 1 : 0.45,
                  cursor: d.showOverlay ? 'pointer' : 'not-allowed',
                }}
                title={
                  d.showOverlay
                    ? showLabels
                      ? 'Hide segmentation labels'
                      : 'Show segmentation labels'
                    : 'Turn annotations on before showing labels'
                }
              >
                T
              </button>
            )}
            <button
              onClick={() => setFullscreen(false)}
              style={{ ...smallBtnStyle, fontSize: 16, minWidth: 32 }}
              title="Exit fullscreen (Esc)"
            >
              ⊗
            </button>
          </div>
        </div>

        {/* Image area */}
        <div
          style={{
            flex: 1, position: 'relative', overflow: 'hidden',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: '#000',
          }}
          onClick={handleImageClick}
          onWheel={handleWheel}
        >
          {loading && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              background: 'rgba(0,0,0,0.6)', zIndex: 3,
              color: 'var(--accent-purple)', fontSize: 28,
            }}>⏳</div>
          )}
          {d.imageBase64 ? (
            <img
              src={`data:image/png;base64,${d.imageBase64}`}
              alt={`Slice ${d.sliceIndex} — ${currentView}`}
              style={{
                maxWidth: '100%', maxHeight: '100%',
                objectFit: 'contain',
                transform: `scale(${zoom})`,
                transformOrigin: 'center',
                imageRendering: 'pixelated',
              }}
              draggable={false}
            />
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>No image loaded</div>
          )}
        </div>

        {/* Footer: navigation */}
        {hasSession && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '8px 16px',
            background: 'rgba(26,27,46,0.95)',
            borderTop: '1px solid var(--border-color)',
            flexShrink: 0,
          }}>
            <button
              onClick={goPrev}
              style={{ ...navBtnStyle, opacity: d.sliceIndex <= 0 ? 0.3 : 1 }}
              disabled={d.sliceIndex <= 0}
            >◀</button>
            <input
              type="range"
              min={0}
              max={Math.max((d.totalSlices || 1) - 1, 0)}
              value={d.sliceIndex || 0}
              onChange={handleSlider}
              style={{ ...sliderStyle, flex: 1 }}
            />
            <button
              onClick={goNext}
              style={{ ...navBtnStyle, opacity: d.sliceIndex >= d.totalSlices - 1 ? 0.3 : 1 }}
              disabled={d.sliceIndex >= d.totalSlices - 1}
            >▶</button>
            <span style={{ ...sliceInfoStyle, flexShrink: 0 }}>
              Slice {(d.sliceIndex || 0) + 1} / {d.totalSlices || '—'}
            </span>
            <div style={{ display: 'flex', gap: 3 }}>
              <button onClick={zoomOut} style={smallBtnStyle}>−</button>
              <button
                onClick={zoomReset}
                style={{
                  ...smallBtnStyle,
                  minWidth: 40,
                  fontSize: 10,
                  color: zoom === 1 ? 'var(--text-muted)' : 'var(--accent-purple)',
                }}
              >
                {Math.round(zoom * 100)}%
              </button>
              <button onClick={zoomIn} style={smallBtnStyle}>+</button>
            </div>
            {d.selectedCoord && (
              <div style={coordBadgeStyle}>
                <span>📍</span>
                <span>({d.selectedCoord.x}, {d.selectedCoord.y})</span>
                <span style={{ opacity: 0.6 }}>S{d.selectedCoord.sliceIndex + 1} · {d.selectedCoord.view}</span>
              </div>
            )}
          </div>
        )}
      </div>,
      document.body,
    )}
    </>
  );
}

export default memo(SliceViewerNode);
