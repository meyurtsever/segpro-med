/**
 * FormatConverterNode
 * ====================
 * Converts between medical image formats (DICOM↔NIfTI↔MAT↔PNG).
 * Corresponds to: POST /api/v1/convert/run
 *
 * WF-1 Steps 3-5: Target Selection → Conversion Engine → Output
 */

import { memo, useCallback } from 'react';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { FormatConverterNodeData } from '../types/nodes';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';

const CONVERSION_TYPES = [
  'DICOM to NIFTI',
  'NIFTI to MAT',
  'DICOM to MAT',
  'NIFTI to PNG',
];

const selectStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-color)',
  borderRadius: 5,
  color: 'var(--text-primary)',
  fontSize: 12,
  outline: 'none',
  cursor: 'pointer',
};

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
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const FORMAT_CONVERTER_INFO: NodeInfo = {
  description:
    'Converts between medical image formats. Supports DICOM→NIfTI, NIfTI→MAT, DICOM→MAT, and NIfTI→PNG conversions. Uses dcm2niix when available, with an automatic Python (pydicom + nibabel) fallback.',
  inputs: [
    'Input path: DICOM directory or NIfTI file (auto-filled from upstream Data Loader)',
    'Conversion type: Select from the dropdown',
    'Output path (optional): Custom location — auto-generated if left empty',
  ],
  outputs: [
    'Converted file at the output path',
    'Output file size',
    'Can be chained to another Format Converter for multi-step conversions',
  ],
  tips: [
    'DICOM → NIfTI: Input must be a directory containing .dcm files',
    'NIfTI → MAT: Preserves volume data, affine transform, and header info',
    'NIfTI → PNG: Creates individual slice images (useful for previews)',
    'Connect a Data Loader upstream to auto-fill the input path',
    'If dcm2niix is not installed, a pure Python fallback is used automatically',
  ],
};

function FormatConverterNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as FormatConverterNodeData;

  const handleTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { conversionType: e.target.value });
    },
    [id, updateNodeData],
  );

  const handleInputPathChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { inputPath: e.target.value });
    },
    [id, updateNodeData],
  );

  const handleOutputPathChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { outputPath: e.target.value });
    },
    [id, updateNodeData],
  );

  return (
    <BaseNode
      nodeId={id}
      nodeType="formatConverter"
      title="Format Converter"
      icon="🔄"
      color="var(--accent-orange)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={FORMAT_CONVERTER_INFO}
    >
      {/* Conversion type selector */}
      <label style={labelStyle}>Conversion Type</label>
      <select
        value={d.conversionType || ''}
        onChange={handleTypeChange}
        style={selectStyle}
      >
        <option value="" disabled>Select conversion…</option>
        {CONVERSION_TYPES.map((t) => (
          <option key={t} value={t}>{t}</option>
        ))}
      </select>

      {/* Input path (can be auto-filled from upstream DataLoaderNode) */}
      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>
          Input Path
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            {' '}(auto from upstream)
          </span>
        </label>
        <input
          type="text"
          value={d.inputPath || ''}
          onChange={handleInputPathChange}
          placeholder="Auto-filled from connected node"
          style={inputStyle}
        />
      </div>

      {!d.inputPath && (
        <NodeHint>
          Connect a Data Loader upstream or enter an input path before running conversion.
        </NodeHint>
      )}

      {/* Output path (optional) */}
      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>
          Output Path
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            {' '}(optional)
          </span>
        </label>
        <input
          type="text"
          value={d.outputPath || ''}
          onChange={handleOutputPathChange}
          placeholder="Auto-generated if empty"
          style={inputStyle}
        />
      </div>

      {/* Result info (shown after successful conversion) */}
      {d.status === 'success' && d.outputPath && (
        <div
          style={{
            marginTop: 8,
            padding: '6px 8px',
            background: 'rgba(76, 175, 139, 0.08)',
            borderRadius: 5,
            border: '1px solid rgba(76, 175, 139, 0.2)',
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--accent-green)', fontWeight: 600 }}>
            ✅ {d.conversionType}
          </div>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-secondary)',
              fontFamily: 'monospace',
              marginTop: 2,
              wordBreak: 'break-all',
            }}
          >
            → {d.outputPath}
          </div>
          {d.outputSizeBytes != null && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              Size: {formatBytes(d.outputSizeBytes)}
            </div>
          )}
        </div>
      )}
    </BaseNode>
  );
}

export default memo(FormatConverterNode);
