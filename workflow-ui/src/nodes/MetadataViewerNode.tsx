/**
 * MetadataViewerNode
 * ==================
 * Terminal Data I/O inspection node for metadata from a loaded session or
 * converted file path. It keeps metadata review reusable instead of requiring
 * users to open the loader node that originally produced the data.
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { MetadataViewerNodeData } from '../types/nodes';
import { buildMetadataRows, truncateMetadataValue } from '../utils/metadataRows';

const METADATA_VIEWER_INFO: NodeInfo = {
  description:
    'Displays metadata from a loaded medical image session or from a connected converted file path. Use it as a terminal inspection node in Data I/O workflows.',
  inputs: [
    'Loaded session from Data Loader',
    'Converted file path from Format Converter',
    'Optional standalone source path configured in the inspector',
  ],
  outputs: [],
  tips: [
    'Connect Data Loader directly when you want metadata immediately after loading.',
    'Connect Format Converter when you want to inspect metadata from the converted output.',
    'Use the filter box to find DICOM tags, NIfTI header fields, or generated orientation metadata.',
  ],
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-color)',
  borderRadius: 5,
  color: 'var(--text-primary)',
  fontSize: 12,
  outline: 'none',
  boxSizing: 'border-box',
};

const summaryRowStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  gap: 6,
  marginBottom: 8,
};

const badgeStyle = (color: string): React.CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'center',
  padding: '2px 8px',
  borderRadius: 12,
  color,
  fontSize: 10,
  fontWeight: 700,
  lineHeight: 1.2,
  textTransform: 'uppercase',
});

const metadataPanelStyle: React.CSSProperties = {
  maxHeight: 190,
  overflowY: 'auto',
  paddingRight: 2,
};

const metadataRowStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '92px minmax(0, 1fr)',
  gap: 8,
  padding: '4px 0',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
  fontSize: 10,
  lineHeight: 1.35,
};

const monoValueStyle: React.CSSProperties = {
  color: 'var(--text-primary)',
  fontFamily: 'monospace',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

function MetadataViewerNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as MetadataViewerNodeData;

  const metadataRows = useMemo(
    () => buildMetadataRows(d.metadata, d.volumeShape),
    [d.metadata, d.volumeShape],
  );
  const filterText = d.filterText || '';
  const filteredRows = useMemo(() => {
    const normalizedFilter = filterText.trim().toLowerCase();
    if (!normalizedFilter) return metadataRows;

    return metadataRows.filter((row) =>
      row.label.toLowerCase().includes(normalizedFilter) ||
      row.value.toLowerCase().includes(normalizedFilter),
    );
  }, [filterText, metadataRows]);

  const handleFilterChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { filterText: event.target.value });
    },
    [id, updateNodeData],
  );

  return (
    <BaseNode
      nodeId={id}
      nodeType="metadataViewer"
      title="Metadata Viewer"
      icon="i"
      color="var(--accent-blue)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={false}
      info={METADATA_VIEWER_INFO}
    >
      {metadataRows.length > 0 ? (
        <>
          <div style={summaryRowStyle}>
            {d.fileType ? (
              <span style={badgeStyle('var(--accent-blue)')}>{d.fileType}</span>
            ) : null}
            <span style={badgeStyle('var(--accent-green)')}>
              {metadataRows.length} fields
            </span>
            {d.volumeShape && d.volumeShape.length > 0 ? (
              <span style={badgeStyle('var(--text-secondary)')}>
                {d.volumeShape.join(' x ')}
              </span>
            ) : null}
          </div>

          <input
            value={filterText}
            onChange={handleFilterChange}
            placeholder="Filter metadata..."
            style={{ ...inputStyle, marginBottom: 8 }}
          />

          {filteredRows.length > 0 ? (
            <div style={metadataPanelStyle}>
              {filteredRows.map((row, index) => (
                <div
                  key={`${row.label}-${index}`}
                  style={{
                    ...metadataRowStyle,
                    borderBottom: index === filteredRows.length - 1
                      ? 'none'
                      : metadataRowStyle.borderBottom,
                  }}
                >
                  <span style={{ color: 'var(--text-muted)', fontWeight: 800 }}>
                    {row.label}
                  </span>
                  <span style={monoValueStyle} title={row.value}>
                    {truncateMetadataValue(row.value, 120)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <NodeHint>No metadata fields match this filter.</NodeHint>
          )}
        </>
      ) : (
        <NodeHint>
          Connect a Data Loader or Format Converter, then run this node to inspect metadata.
        </NodeHint>
      )}

      {d.sourcePath ? (
        <div
          style={{
            marginTop: 8,
            color: 'var(--text-muted)',
            fontSize: 10,
            fontFamily: 'monospace',
            wordBreak: 'break-all',
          }}
        >
          {d.sourcePath}
        </div>
      ) : null}
    </BaseNode>
  );
}

export default memo(MetadataViewerNode);
