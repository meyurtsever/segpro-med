/**
 * FormatConverterNode
 * ====================
 * Converts between medical image formats (DICOM↔NIfTI↔MAT↔PNG).
 * Corresponds to: POST /api/v1/convert/run
 *
 * WF-1 Steps 3-5: Target Selection → Conversion Engine → Output
 */

import { memo, useCallback, useState } from 'react';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { FormatConverterNodeData } from '../types/nodes';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';
import * as api from '../api/client';

const CONVERSION_TYPES = [
  'DICOM to NIFTI',
  'NIFTI to MAT',
  'DICOM to MAT',
  'NIFTI to PNG',
];

const AXIS_OPTIONS = [
  { value: 2, label: 'Axial (axis 2)' },
  { value: 1, label: 'Coronal (axis 1)' },
  { value: 0, label: 'Sagittal (axis 0)' },
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

const checkboxRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginTop: 8,
  color: 'var(--text-secondary)',
  fontSize: 11,
};

const actionButtonStyle = (active = false): React.CSSProperties => ({
  width: '100%',
  padding: '7px 9px',
  borderRadius: 5,
  border: `1px solid ${active ? 'var(--accent-blue)' : 'rgba(79, 141, 245, 0.4)'}`,
  background: active ? 'rgba(79, 141, 245, 0.18)' : 'rgba(79, 141, 245, 0.1)',
  color: 'var(--accent-blue)',
  fontSize: 11,
  fontWeight: 800,
  cursor: active ? 'progress' : 'pointer',
  marginTop: 8,
});

const previewFrameStyle: React.CSSProperties = {
  marginTop: 8,
  width: '100%',
  aspectRatio: '1',
  background: '#000',
  borderRadius: 5,
  overflow: 'hidden',
  border: '1px solid var(--border-color)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};

const previewImageStyle: React.CSSProperties = {
  maxWidth: '100%',
  maxHeight: '100%',
  objectFit: 'contain',
  imageRendering: 'pixelated',
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getPathGuidance(conversionType: string | undefined, inputPath: string | undefined) {
  const normalizedPath = inputPath?.trim().toLowerCase();
  if (!conversionType || !normalizedPath) return undefined;

  if (conversionType.startsWith('NIFTI') &&
    !normalizedPath.endsWith('.nii') &&
    !normalizedPath.endsWith('.nii.gz')
  ) {
    return 'This conversion expects a NIfTI file ending in .nii or .nii.gz.';
  }

  if (conversionType.startsWith('DICOM') &&
    (normalizedPath.endsWith('.nii') || normalizedPath.endsWith('.nii.gz') || normalizedPath.endsWith('.mat'))
  ) {
    return 'This conversion expects a DICOM directory or a single .dcm file.';
  }

  return undefined;
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
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

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

  const handleAxisChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { axis: Number(e.target.value) });
    },
    [id, updateNodeData],
  );

  const handleCompressChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { compress: e.target.checked });
    },
    [id, updateNodeData],
  );

  const pathGuidance = getPathGuidance(d.conversionType, d.inputPath);
  const canPreviewOutput = Boolean(d.outputPath) && d.conversionType !== 'NIFTI to PNG';

  const handlePreviewOutput = useCallback(async () => {
    if (!d.outputPath || !canPreviewOutput) return;

    setPreviewLoading(true);
    setPreviewError(null);

    try {
      const loaded = await api.loadDataFromPath(d.outputPath);
      const slice = await api.getSlice(loaded.session_id, 0, 'axial');
      updateNodeData(id, {
        previewSessionId: loaded.session_id,
        previewFileType: loaded.file_type,
        previewImageBase64: slice.image_base64,
        previewTotalSlices: slice.total_slices,
        previewVolumeShape: loaded.volume_shape,
        previewMetadata: loaded.metadata,
      });
    } catch (error) {
      setPreviewError(error instanceof Error
        ? error.message
        : 'Converted output could not be previewed.');
    } finally {
      setPreviewLoading(false);
    }
  }, [canPreviewOutput, d.outputPath, id, updateNodeData]);

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

      {pathGuidance ? (
        <NodeHint
          style={{
            color: 'var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.08)',
            borderColor: 'rgba(224, 92, 92, 0.25)',
          }}
        >
          {pathGuidance}
        </NodeHint>
      ) : null}

      {d.conversionType === 'DICOM to NIFTI' ? (
        <label style={checkboxRowStyle}>
          <input
            type="checkbox"
            checked={d.compress ?? true}
            onChange={handleCompressChange}
          />
          Compress NIfTI output (.nii.gz)
        </label>
      ) : null}

      {d.conversionType === 'NIFTI to PNG' ? (
        <div style={{ marginTop: 8 }}>
          <label style={labelStyle}>PNG Slice Axis</label>
          <select
            value={d.axis ?? 2}
            onChange={handleAxisChange}
            style={selectStyle}
          >
            {AXIS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      ) : null}

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

      {d.status === 'success' && d.outputPath ? (
        canPreviewOutput ? (
          <>
            <button
              type="button"
              onClick={handlePreviewOutput}
              disabled={previewLoading}
              style={actionButtonStyle(previewLoading)}
            >
              {previewLoading ? 'Loading Preview...' : 'Preview Converted Output'}
            </button>
            {previewError ? (
              <NodeHint
                style={{
                  color: 'var(--accent-red)',
                  background: 'rgba(224, 92, 92, 0.08)',
                  borderColor: 'rgba(224, 92, 92, 0.25)',
                }}
              >
                {previewError}
              </NodeHint>
            ) : null}
            {d.previewImageBase64 ? (
              <>
                <div style={previewFrameStyle}>
                  <img
                    src={`data:image/png;base64,${d.previewImageBase64}`}
                    alt="Converted output preview"
                    style={previewImageStyle}
                  />
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  {d.previewFileType?.toUpperCase() || 'OUTPUT'}
                  {d.previewTotalSlices ? ` - ${d.previewTotalSlices} slices` : ''}
                  {d.previewVolumeShape ? ` - ${d.previewVolumeShape.join(' x ')}` : ''}
                </div>
              </>
            ) : null}
          </>
        ) : (
          <NodeHint>
            PNG series outputs are written as folders. Connect this node to a Metadata Viewer
            or inspect the output directory from the filesystem.
          </NodeHint>
        )
      ) : null}
    </BaseNode>
  );
}

export default memo(FormatConverterNode);
