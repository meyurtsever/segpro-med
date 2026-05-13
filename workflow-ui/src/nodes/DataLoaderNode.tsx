/**
 * DataLoaderNode
 * ===============
 * Loads DICOM/NIfTI/MAT data from a local filesystem path.
 * Supports ICD-10 / keyword patient search via the retrieval system.
 * Corresponds to: POST /api/v1/data/load
 *
 * WF-1 Step 1: File Input + Format Detect
 */

import { memo, useCallback, useState, useEffect, useRef } from 'react';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { DataLoaderNodeData } from '../types/nodes';
import FilePickerModal from '../components/FilePickerModal';
import type { NodeInfo } from '../components/InfoModal';
import * as api from '../api/client';
import type { PatientSearchResult } from '../api/client';

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

function DataLoaderNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as DataLoaderNodeData;
  const [showPicker, setShowPicker] = useState(false);

  // ── Patient search state ──────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<PatientSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
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
      updateNodeData(id, { path: e.target.value });
    },
    [id, updateNodeData],
  );

  // Debounced patient search
  useEffect(() => {
    if (searchQuery.trim().length < 2) {
      setSearchResults([]);
      setShowDropdown(false);
      return;
    }
    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await api.searchPatients(searchQuery.trim(), d.searchRoot || undefined);
        setSearchResults(res.results);
        setShowDropdown(true);
      } catch {
        setSearchResults([]);
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
      setSearchQuery('');
      setShowDropdown(false);
    },
    [id, updateNodeData],
  );

  const inspectUrl = d.sessionId
    ? `http://localhost:7860?session_id=${d.sessionId}&tab=viewer`
    : undefined;

  return (
    <BaseNode
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
        🔍 Patient Search
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
            <input
              type="text"
              value={d.searchRoot || ''}
              onChange={(e) => updateNodeData(id, { searchRoot: e.target.value })}
              placeholder={serverRoot || 'e.g. C:\\data\\500 MR'}
              style={{ ...inputStyle, fontFamily: 'inherit', fontSize: 11 }}
            />
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
      <div style={{ display: 'flex', gap: 4 }}>
        <input
          type="text"
          value={d.path || 'C://Users//Yurtsever//Downloads//segpro-med//cvm_48_t1'}
          onChange={handlePathChange}
          placeholder="C:\data\patient01 or /data/brain.nii.gz"
          style={{ ...inputStyle, flex: 1 }}
        />
        <button
          onClick={() => setShowPicker(true)}
          style={{
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-color)',
            borderRadius: 5,
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: '6px 8px',
            fontSize: 14,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
          }}
          title="Browse filesystem"
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

      {/* Session info (shown after successful load) */}
      {d.sessionId && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
            <span style={badgeStyle('var(--accent-green)')}>
              {d.fileType?.toUpperCase()}
            </span>
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

      {/* File Picker Modal */}
      <FilePickerModal
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        onSelect={(selectedPath) => {
          updateNodeData(id, { path: selectedPath });
          setShowPicker(false);
        }}
        initialPath={d.path || ''}
      />
    </BaseNode>
  );
}

export default memo(DataLoaderNode);

