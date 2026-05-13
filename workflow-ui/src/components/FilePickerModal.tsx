/**
 * FilePickerModal — Browse the server filesystem to select a file or directory.
 * Communicates with GET /api/v1/fs/browse.
 */

import { useState, useEffect, useCallback, memo } from 'react';
import { browsePath, type BrowseEntry } from '../api/client';

interface FilePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  title?: string;
  initialPath?: string;
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
  width: 520,
  maxHeight: '80vh',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
  overflow: 'hidden',
};

function formatSize(size: number | null): string {
  if (size === null) return '';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function getEntryIcon(entry: BrowseEntry): string {
  if (entry.is_dir) return '📁';
  const ext = entry.extension.toLowerCase();
  const name = entry.name.toLowerCase();
  if (ext === '.dcm') return '🏥';
  if (ext === '.nii' || name.endsWith('.nii.gz')) return '🧠';
  if (ext === '.mat') return '📊';
  if (['.png', '.jpg', '.jpeg', '.bmp', '.tiff'].includes(ext)) return '🖼️';
  if (ext === '.zip') return '📦';
  return '📄';
}

function FilePickerModal({
  isOpen,
  onClose,
  onSelect,
  title = 'Select File or Directory',
  initialPath = '',
}: FilePickerModalProps) {
  const [currentPath, setCurrentPath] = useState('/');
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<BrowseEntry | null>(null);
  const [pathInput, setPathInput] = useState('');

  const loadDirectory = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    setSelectedEntry(null);
    try {
      const res = await browsePath(path);
      setCurrentPath(res.path);
      setParentPath(res.parent);
      setEntries(res.entries);
      setPathInput(res.path);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to browse directory');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadDirectory(initialPath || '');
    }
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleEntryClick = (entry: BrowseEntry) => {
    setSelectedEntry(entry);
  };

  const handleEntryDoubleClick = (entry: BrowseEntry) => {
    if (entry.is_dir) {
      loadDirectory(entry.path);
    } else {
      onSelect(entry.path);
      onClose();
    }
  };

  const handleSelect = () => {
    if (selectedEntry) {
      onSelect(selectedEntry.path);
      onClose();
    } else if (currentPath && currentPath !== '/') {
      onSelect(currentPath);
      onClose();
    }
  };

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pathInput) loadDirectory(pathInput);
  };

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div
          style={{
            padding: '14px 20px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <span style={{ fontSize: 20 }}>📂</span>
          <span
            style={{
              flex: 1,
              fontWeight: 700,
              fontSize: 14,
              color: 'var(--text-primary)',
            }}
          >
            {title}
          </span>
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
          >
            ✕
          </button>
        </div>

        {/* Path bar */}
        <form
          onSubmit={handlePathSubmit}
          style={{
            padding: '10px 20px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            gap: 8,
          }}
        >
          {parentPath && (
            <button
              type="button"
              onClick={() => loadDirectory(parentPath)}
              style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)',
                padding: '6px 10px',
                borderRadius: 5,
                cursor: 'pointer',
                fontSize: 14,
                flexShrink: 0,
              }}
              title="Go to parent directory"
            >
              ⬆
            </button>
          )}
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            style={{
              flex: 1,
              padding: '6px 10px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: 5,
              color: 'var(--text-primary)',
              fontSize: 12,
              fontFamily: 'monospace',
              outline: 'none',
            }}
            placeholder="Enter path..."
          />
          <button
            type="submit"
            style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-secondary)',
              padding: '6px 12px',
              borderRadius: 5,
              cursor: 'pointer',
              fontSize: 12,
              flexShrink: 0,
            }}
          >
            Go
          </button>
        </form>

        {/* File list */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '4px 0',
            minHeight: 200,
            maxHeight: 400,
          }}
        >
          {loading && (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading…
            </div>
          )}

          {error && (
            <div
              style={{
                padding: '12px 20px',
                color: 'var(--accent-red)',
                fontSize: 12,
                background: 'rgba(224, 92, 92, 0.08)',
              }}
            >
              {error}
            </div>
          )}

          {!loading && !error && entries.length === 0 && (
            <div
              style={{
                padding: 20,
                textAlign: 'center',
                color: 'var(--text-muted)',
                fontSize: 12,
              }}
            >
              Empty directory
            </div>
          )}

          {!loading &&
            entries.map((entry) => (
              <div
                key={entry.path}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => handleEntryDoubleClick(entry)}
                style={{
                  padding: '8px 20px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  cursor: 'pointer',
                  background:
                    selectedEntry?.path === entry.path
                      ? 'rgba(79, 141, 245, 0.1)'
                      : 'transparent',
                  borderLeft:
                    selectedEntry?.path === entry.path
                      ? '3px solid var(--accent-blue)'
                      : '3px solid transparent',
                  transition: 'all 0.1s ease',
                }}
                onMouseEnter={(e) => {
                  if (selectedEntry?.path !== entry.path) {
                    e.currentTarget.style.background = 'var(--bg-tertiary)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedEntry?.path !== entry.path) {
                    e.currentTarget.style.background =
                      selectedEntry?.path === entry.path
                        ? 'rgba(79, 141, 245, 0.1)'
                        : 'transparent';
                  }
                }}
              >
                <span style={{ fontSize: 16, flexShrink: 0 }}>{getEntryIcon(entry)}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 13,
                      color: entry.is_dir ? 'var(--accent-blue)' : 'var(--text-primary)',
                      fontWeight: entry.is_dir ? 600 : 400,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {entry.name}
                  </div>
                </div>
                {entry.size !== null && (
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0 }}>
                    {formatSize(entry.size)}
                  </span>
                )}
                {entry.is_dir && (
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0 }}>
                    →
                  </span>
                )}
              </div>
            ))}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 260 }}>
            {selectedEntry
              ? `Selected: ${selectedEntry.name}`
              : 'Double-click folder to enter · Select item then click "Select"'}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={onClose}
              style={{
                padding: '6px 16px',
                borderRadius: 5,
                border: '1px solid var(--border-color)',
                background: 'transparent',
                color: 'var(--text-secondary)',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!selectedEntry && currentPath === '/'}
              style={{
                padding: '6px 16px',
                borderRadius: 5,
                border: '1px solid var(--accent-blue)',
                background: 'var(--accent-blue)',
                color: 'white',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                opacity: !selectedEntry && currentPath === '/' ? 0.5 : 1,
              }}
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default memo(FilePickerModal);
