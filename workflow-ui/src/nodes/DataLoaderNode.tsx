/**
 * DataLoaderNode
 * ===============
 * Loads DICOM/NIfTI/MAT data from a local filesystem path.
 * Supports ICD-10 / keyword patient search via the retrieval system.
 * Corresponds to: POST /api/v1/data/load
 *
 * WF-1 Step 1: File Input + Format Detect
 */

import { memo, useCallback, useMemo, useState, useEffect, useRef } from 'react';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { DataLoaderNodeData } from '../types/nodes';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';
import * as api from '../api/client';
import type { PatientSearchResult } from '../api/client';
import { buildMetadataRows, truncateMetadataValue } from '../utils/metadataRows';
import {
  TASK_DATA_AUTO_LOAD_EVENT,
  supportsTaskDataAutoLoad,
} from '../engine/taskEvents';

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-color)',
  borderRadius: 5,
  color: 'var(--text-primary)',
  fontSize: 12,
  fontFamily: 'monospace',
  outline: 'none',
  boxSizing: 'border-box',
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
};

const fieldWithButtonsStyle: React.CSSProperties = {
  display: 'flex',
  gap: 4,
  alignItems: 'stretch',
};

const browseButtonStyle = (active = false): React.CSSProperties => ({
  background: active ? 'rgba(79, 141, 245, 0.16)' : 'var(--bg-tertiary)',
  border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border-color)'}`,
  borderRadius: 5,
  color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
  cursor: active ? 'progress' : 'pointer',
  padding: '6px 8px',
  fontSize: 11,
  fontWeight: 700,
  flexShrink: 0,
  minWidth: 38,
});

const badgeStyle = (color: string): React.CSSProperties => ({
  display: 'inline-block',
  padding: '2px 8px',
  borderRadius: 12,
  fontSize: 10,
  fontWeight: 600,
  background: `${color}20`,
  color,
  border: `1px solid ${color}40`,
});

const metadataToggleStyle = (expanded: boolean): React.CSSProperties => ({
  appearance: 'none',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '2px 8px',
  borderRadius: 12,
  border: '1px solid transparent',
  background: 'transparent',
  color: 'var(--accent-blue)',
  fontSize: 10,
  fontWeight: 600,
  lineHeight: 1.2,
  opacity: expanded ? 1 : 0.92,
  cursor: 'pointer',
  textTransform: 'uppercase',
});

const metadataPanelStyle: React.CSSProperties = {
  marginTop: 6,
  maxHeight: 150,
  overflowY: 'auto',
  paddingRight: 2,
};

const metadataRowStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '86px minmax(0, 1fr)',
  gap: 8,
  padding: '4px 0',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
  fontSize: 10,
  lineHeight: 1.35,
};

const DATA_LOADER_INFO: NodeInfo = {
  description:
    'Loads medical imaging data (DICOM series, NIfTI volumes, or MATLAB files) from a local filesystem path and creates a processing session for downstream nodes.',
  inputs: [
    'Search by ICD-10 code (e.g. "D18.02") or clinical keyword (e.g. "glioblastoma", "cavernoma")',
    'Filesystem path to a DICOM directory, NIfTI file (.nii, .nii.gz), or MATLAB file (.mat)',
    'Use the 📁 button to browse and select files/folders visually',
  ],
  outputs: [
    'Session ID for referencing loaded data in downstream nodes',
    'Detected file type (DICOM, NIfTI, MAT)',
    'Volume shape (depth × height × width)',
    'DICOM metadata (Modality, Patient info, Series Description, etc.)',
  ],
  tips: [
    'Type an ICD-10 code or condition name in the search box to find matching patients',
    'For DICOM data, point to the directory containing all .dcm files',
    'NIfTI files can be .nii or compressed .nii.gz',
    'Connect to a Format Converter to auto-fill its input path',
    'Use "Inspect on Tool" to view the loaded data in the Gradio viewer',
  ],
};

function requestTaskDataAutoLoad(
  nodeId: string,
  nodeData: Record<string, unknown>,
  path: string,
) {
  if (typeof window === 'undefined') return;
  if (!supportsTaskDataAutoLoad(nodeData.taskTemplateId)) return;
  if (nodeData.taskNodeKey !== 'loader') return;
  if (!path.trim()) return;

  window.dispatchEvent(new CustomEvent(TASK_DATA_AUTO_LOAD_EVENT, {
    detail: {
      nodeId,
      templateId: nodeData.taskTemplateId,
      path,
    },
  }));
}

function DataLoaderNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as DataLoaderNodeData;
  const [pickerBusy, setPickerBusy] = useState<string | null>(null);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const [showMetadata, setShowMetadata] = useState(false);

  // ── Patient search state ──────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<PatientSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [showRootField, setShowRootField] = useState(false);
  const [serverRoot, setServerRoot] = useState<string>('');
  const [rootStatus, setRootStatus] = useState<'unknown' | 'ok' | 'missing'>('unknown');
  const searchWrapperRef = useRef<HTMLDivElement>(null);

  // Load the server's current root directory once on mount
  useEffect(() => {
    api.getPatientRoot().then((r) => {
      setServerRoot(r.root_directory);
      setRootStatus(r.directory_exists ? 'ok' : 'missing');
    }).catch(() => {
      /* API not running yet — fail silently */
    });
  }, []);

  const handlePathChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setPickerError(null);
      updateNodeData(id, { path: e.target.value });
    },
    [id, updateNodeData],
  );

  const handleNativePathPick = useCallback(
    async (target: 'pathFile' | 'pathDirectory' | 'searchRoot') => {
      const mode = target === 'pathFile' ? 'file' : 'directory';
      const initialPath = target === 'searchRoot'
        ? d.searchRoot || serverRoot
        : d.path;

      setPickerBusy(target);
      setPickerError(null);

      try {
        const result = await api.openNativePathDialog(mode, initialPath || '');
        if (result.cancelled || !result.path) return;

        if (target === 'searchRoot') {
          updateNodeData(id, { searchRoot: result.path });
        } else {
          updateNodeData(id, { path: result.path });
          requestTaskDataAutoLoad(id, data as Record<string, unknown>, result.path);
        }
      } catch (err) {
        setPickerError(
          err instanceof Error
            ? err.message
            : 'Native file selector could not be opened',
        );
      } finally {
        setPickerBusy(null);
      }
    },
    [d.path, d.searchRoot, data, id, serverRoot, updateNodeData],
  );

  // Debounced patient search
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
        const res = await api.searchPatients(searchQuery.trim(), d.searchRoot || undefined);
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
  }, [searchQuery, d.searchRoot]);

  // Close dropdown when clicking outside the search wrapper
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (searchWrapperRef.current && !searchWrapperRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  const handleResultSelect = useCallback(
    (result: PatientSearchResult) => {
      updateNodeData(id, { path: result.path });
      requestTaskDataAutoLoad(id, data as Record<string, unknown>, result.path);
      setSearchQuery('');
      setShowDropdown(false);
    },
    [data, id, updateNodeData],
  );

  const inspectUrl = d.sessionId
    ? `http://localhost:7860?session_id=${d.sessionId}&tab=viewer`
    : undefined;
  const metadataRows = useMemo(
    () => buildMetadataRows(d.metadata, d.volumeShape),
    [d.metadata, d.volumeShape],
  );
  const hasMetadata = metadataRows.length > 0;

  return (
    <BaseNode
      nodeId={id}
      nodeType="dataLoader"
      title="Data Loader"
      icon="📂"
      color="var(--accent-blue)"
      status={d.status}
      error={d.error}
      hasInput={false}
      hasOutput={true}
      inspectUrl={inspectUrl}
      info={DATA_LOADER_INFO}
    >
      {/* ── Patient search (primary) ─────────────────────────────────────── */}
      <label style={labelStyle}>
        Patient Search
      </label>
      <div ref={searchWrapperRef} style={{ position: 'relative' }}>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="ICD-10 code or keyword — e.g. 'D18.02', 'glioblastoma', 'cavernoma'…"
          style={{
            ...inputStyle,
            fontFamily: 'inherit',
            paddingRight: isSearching ? 28 : 8,
          }}
          onFocus={() => searchResults.length > 0 && setShowDropdown(true)}
        />

        {/* Spinner */}
        {isSearching && (
          <span
            style={{
              position: 'absolute',
              right: 8,
              top: '50%',
              transform: 'translateY(-50%)',
              fontSize: 11,
              color: 'var(--text-muted)',
              pointerEvents: 'none',
            }}
          >
            ⏳
          </span>
        )}

        {/* Results dropdown */}
        {showDropdown && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              zIndex: 200,
              background: 'var(--bg-secondary, #1e1e2e)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              marginTop: 2,
              maxHeight: 220,
              overflowY: 'auto',
              boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            }}
          >
            {searchResults.length === 0 ? (
              <div
                style={{
                  padding: '8px 10px',
                  fontSize: 11,
                  color: 'var(--text-muted)',
                  fontStyle: 'italic',
                }}
              >
                No patients found — try a different term or enter the path directly below.
              </div>
            ) : (
              searchResults.map((r) => (
                <button
                  key={r.display_name}
                  onClick={() => handleResultSelect(r)}
                  style={{
                    display: 'block',
                    width: '100%',
                    background: 'transparent',
                    border: 'none',
                    borderBottom: '1px solid var(--border-color)',
                    cursor: 'pointer',
                    padding: '7px 10px',
                    textAlign: 'left',
                    color: 'var(--text-primary)',
                  }}
                  onMouseEnter={(e) =>
                    ((e.currentTarget as HTMLElement).style.background =
                      'var(--bg-tertiary)')
                  }
                  onMouseLeave={(e) =>
                    ((e.currentTarget as HTMLElement).style.background = 'transparent')
                  }
                >
                  <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 2 }}>
                    {r.display_name}
                  </div>
                  <div style={{ display: 'flex', gap: 5, alignItems: 'center', flexWrap: 'wrap' }}>
                    {r.icd10_code && (
                      <span style={badgeStyle('var(--accent-blue)')}>
                        {r.icd10_code}
                      </span>
                    )}
                    {r.icd10_description && (
                      <span
                        style={{
                          fontSize: 10,
                          color: 'var(--text-muted)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: 200,
                        }}
                      >
                        {r.icd10_description}
                      </span>
                    )}
                  </div>
                  {r.path && (
                    <div
                      style={{
                        fontSize: 10,
                        color: 'var(--text-muted)',
                        fontFamily: 'monospace',
                        marginTop: 2,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {r.path}
                    </div>
                  )}
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {searchError && (
        <div
          style={{
            marginTop: 4,
            color: 'var(--accent-red)',
            fontSize: 10,
            lineHeight: 1.4,
          }}
        >
          {searchError}
        </div>
      )}

      {/* Search root (advanced, collapsed by default) */}
      <div style={{ marginTop: 4 }}>
        <button
          onClick={() => setShowRootField((v) => !v)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--text-muted)',
            fontSize: 10,
            padding: 0,
          }}
        >
          {showRootField ? '▾' : '▸'} Dataset root directory
          {rootStatus === 'ok' && (
            <span style={{ marginLeft: 5, color: 'var(--accent-green)', fontSize: 10 }}>✓ found</span>
          )}
          {rootStatus === 'missing' && (
            <span style={{ marginLeft: 5, color: 'var(--accent-red, #f38ba8)', fontSize: 10 }}>⚠ not found</span>
          )}
        </button>
        {showRootField && (
          <>
            <div style={{ height: 4 }} />
            {serverRoot && !d.searchRoot && (
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Server default: {serverRoot}
              </div>
            )}
            <div style={fieldWithButtonsStyle}>
              <input
                type="text"
                value={d.searchRoot || ''}
                onChange={(e) => updateNodeData(id, { searchRoot: e.target.value })}
                placeholder={serverRoot || 'e.g. C:\\data\\500 MR'}
                style={{ ...inputStyle, fontFamily: 'inherit', fontSize: 11, flex: 1 }}
              />
              <button
                type="button"
                onClick={() => handleNativePathPick('searchRoot')}
                style={browseButtonStyle(pickerBusy === 'searchRoot')}
                disabled={pickerBusy !== null}
                title="Select dataset root directory"
              >
                Dir
              </button>
            </div>
          </>
        )}
      </div>

      {/* Divider */}
      <div
        style={{
          margin: '10px 0 8px',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span style={{ fontSize: 10, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
          or enter path directly
        </span>
        <div style={{ flex: 1, borderTop: '1px solid var(--border-color)' }} />
      </div>

      {/* Path input with file picker */}
      <label style={labelStyle}>File / Directory Path</label>
      <div style={fieldWithButtonsStyle}>
        <input
          type="text"
          value={d.path || ''}
          onChange={handlePathChange}
          placeholder="C:\data\patient01 or /data/brain.nii.gz"
          style={{ ...inputStyle, flex: 1 }}
        />
        <button
          type="button"
          onClick={() => handleNativePathPick('pathFile')}
          style={browseButtonStyle(pickerBusy === 'pathFile')}
          disabled={pickerBusy !== null}
          title="Select medical image file"
        >
          File
        </button>
        <button
          type="button"
          onClick={() => handleNativePathPick('pathDirectory')}
          style={browseButtonStyle(pickerBusy === 'pathDirectory')}
          disabled={pickerBusy !== null}
          title="Select DICOM or dataset directory"
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--accent-blue)';
            (e.target as HTMLElement).style.color = 'var(--accent-blue)';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--border-color)';
            (e.target as HTMLElement).style.color = 'var(--text-secondary)';
          }}
        >
          📁
        </button>
      </div>

      {pickerError && (
        <NodeHint
          style={{
            color: 'var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.08)',
            borderColor: 'rgba(224, 92, 92, 0.25)',
          }}
        >
          {pickerError}
        </NodeHint>
      )}

      {/* Session info (shown after successful load) */}
      {d.sessionId && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
            <span style={badgeStyle('var(--accent-green)')}>
              {(d.fileType || 'loaded').toUpperCase()}
            </span>
            {hasMetadata ? (
              <button
                type="button"
                onClick={() => setShowMetadata((current) => !current)}
                style={metadataToggleStyle(showMetadata)}
                title="Show loaded image metadata"
                aria-expanded={showMetadata}
              >
                Metadata {showMetadata ? '^' : 'v'}
              </button>
            ) : null}
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
              {d.sessionId}
            </span>
          </div>

          {d.volumeShape && (
            <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
              Shape: [{d.volumeShape.join(' × ')}]
            </div>
          )}
        </div>
      )}

      {d.sessionId && hasMetadata && showMetadata ? (
        <div style={metadataPanelStyle}>
          {metadataRows.map((row, index) => (
            <div
              key={`${row.label}-${index}`}
              style={{
                ...metadataRowStyle,
                borderBottom: index === metadataRows.length - 1
                  ? 'none'
                  : metadataRowStyle.borderBottom,
              }}
            >
              <span style={{ color: 'var(--text-muted)', fontWeight: 800 }}>
                {row.label}
              </span>
              <span
                style={{
                  color: 'var(--text-primary)',
                  fontFamily: 'monospace',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={row.value}
              >
                {truncateMetadataValue(row.value)}
              </span>
            </div>
          ))}
        </div>
      ) : null}

    </BaseNode>
  );
}

export default memo(DataLoaderNode);

