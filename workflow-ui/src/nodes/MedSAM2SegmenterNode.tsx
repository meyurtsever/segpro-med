/**
 * MedSAM2SegmenterNode
 * ====================
 * Automation-oriented SAM2 node. Interactive users should usually run SAM2
 * from Interactive Annotator on the visible slice; this node is for pre-running
 * SAM2 on one slice, a range of slices, or the whole selected view.
 */

import { memo, useCallback, useEffect, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { MedSAM2SegmenterNodeData, SegmentationRunMode } from '../types/nodes';
import * as api from '../api/client';

const MEDSAM2_INFO: NodeInfo = {
  description:
    'Runs SAM2 automatically for workflow automation. For manual, visual slice selection, run SAM2 from the Interactive Annotator AI Segmentation panel instead.',
  inputs: [
    'Loaded session from Data Loader',
    'Converted file path from Format Converter',
    'Optional profile from Segmentation Profile',
  ],
  outputs: [
    'Segmentation result containing editable polygon shapes and per-slice batch details',
  ],
  tips: [
    'Single slice mode is useful for deterministic pre-processing.',
    'Range and whole scan modes can be slow because SAM2 runs once per slice.',
    'Connect the output to Interactive Annotator to refine generated polygons.',
  ],
};

const VIEWS = ['axial', 'coronal', 'sagittal'] as const;
const RUN_MODES: Array<{ value: SegmentationRunMode; label: string }> = [
  { value: 'single', label: 'Single slice' },
  { value: 'range', label: 'Slice range' },
  { value: 'wholeVolume', label: 'Whole scan' },
];

const PROMPT_MODES = [
  { value: 'auto', label: 'Automatic masks' },
  { value: 'prompt', label: 'Point / box prompt' },
] as const;

const FALLBACK_CONFIGS = [
  'fast',
  'balanced',
  'high_detail',
  'small_structures',
  'tumor_detection',
  'skull_stripping',
  'mammography',
];

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
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

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
};

const fieldGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 76px',
  gap: 6,
  marginTop: 8,
};

const rangeGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr 72px',
  gap: 6,
  marginTop: 8,
};

const summaryStyle: React.CSSProperties = {
  marginTop: 8,
  padding: '7px 8px',
  borderRadius: 6,
  border: '1px solid rgba(76, 175, 139, 0.24)',
  background: 'rgba(76, 175, 139, 0.08)',
  fontSize: 11,
  lineHeight: 1.45,
};

const metricRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 8,
  color: 'var(--text-secondary)',
};

function titleCase(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function toNumber(value: string, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : fallback;
}

function MedSAM2SegmenterNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as MedSAM2SegmenterNodeData;
  const [loadingConfigs, setLoadingConfigs] = useState(() => d.availableConfigs.length === 0);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    if (d.availableConfigs.length > 0) return;

    api.getSegmentationConfigs()
      .then((res) => {
        setConfigError(null);
        updateNodeData(id, {
          availableConfigs: res.configs,
        } as Partial<MedSAM2SegmenterNodeData>);
      })
      .catch((error) => {
        setConfigError(error instanceof Error
          ? error.message
          : 'Failed to load segmentation configs');
      })
      .finally(() => setLoadingConfigs(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfigChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { configName: event.target.value } as Partial<MedSAM2SegmenterNodeData>);
    },
    [id, updateNodeData],
  );

  const handleRunModeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { runMode: event.target.value as SegmentationRunMode } as Partial<MedSAM2SegmenterNodeData>);
    },
    [id, updateNodeData],
  );

  const handlePromptModeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { promptMode: event.target.value } as Partial<MedSAM2SegmenterNodeData>);
    },
    [id, updateNodeData],
  );

  const handleViewChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { view: event.target.value } as Partial<MedSAM2SegmenterNodeData>);
    },
    [id, updateNodeData],
  );

  const updateNumberField = useCallback(
    (key: keyof MedSAM2SegmenterNodeData, value: string, fallback = 0) => {
      updateNodeData(id, { [key]: toNumber(value, fallback) } as Partial<MedSAM2SegmenterNodeData>);
    },
    [id, updateNodeData],
  );

  const result = d.segmentationResult;
  const hasSession = Boolean(d.sessionId);
  const configOptions = d.availableConfigs.length > 0
    ? d.availableConfigs.map((config) => config.name)
    : FALLBACK_CONFIGS;
  const runMode = d.runMode || 'single';
  const promptMode = d.promptMode || (d.segmentationPrompt ? 'prompt' : 'auto');
  const promptSummary = d.segmentationPrompt
    ? `${d.segmentationPrompt.points.length} point(s), ${d.segmentationPrompt.boxes.length} box(es)`
    : 'No prompt connected';

  return (
    <BaseNode
      nodeId={id}
      nodeType="medsam2Segmenter"
      title="Batch SAM2 Segmenter"
      icon="S2"
      color="var(--accent-purple)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={MEDSAM2_INFO}
    >
      <div style={{ fontSize: 11, color: hasSession ? 'var(--accent-green)' : 'var(--text-muted)' }}>
        {hasSession ? 'Data session connected' : 'Awaiting Data Loader or Format Converter'}
      </div>

      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>Mode</label>
        <select value={promptMode} onChange={handlePromptModeChange} style={selectStyle}>
          {PROMPT_MODES.map((mode) => (
            <option key={mode.value} value={mode.value}>{mode.label}</option>
          ))}
        </select>
      </div>

      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>Run Mode</label>
        <select value={runMode} onChange={handleRunModeChange} style={selectStyle}>
          {RUN_MODES.map((mode) => (
            <option key={mode.value} value={mode.value}>{mode.label}</option>
          ))}
        </select>
      </div>

      {promptMode === 'prompt' ? (
        <NodeHint>
          {promptSummary}. Create or select a point/rectangle in Interactive Annotator, run Prompt SAM2 there, or connect the annotator prompt output here.
        </NodeHint>
      ) : null}

      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>Segmentation Profile</label>
        <select
          value={d.configName || 'fast'}
          onChange={handleConfigChange}
          style={selectStyle}
          disabled={loadingConfigs}
        >
          {configOptions.map((configName) => (
            <option key={configName} value={configName}>
              {titleCase(configName)}
            </option>
          ))}
        </select>
      </div>

      <div style={fieldGridStyle}>
        <div>
          <label style={labelStyle}>View</label>
          <select value={d.view || 'axial'} onChange={handleViewChange} style={selectStyle}>
            {VIEWS.map((view) => (
              <option key={view} value={view}>{titleCase(view)}</option>
            ))}
          </select>
        </div>
        {runMode === 'single' ? (
          <div>
            <label style={labelStyle}>Slice</label>
            <input
              type="number"
              min={0}
              value={Number(d.sliceIndex ?? 0)}
              onChange={(event) => updateNumberField('sliceIndex', event.target.value)}
              style={inputStyle}
            />
          </div>
        ) : (
          <div>
            <label style={labelStyle}>Step</label>
            <input
              type="number"
              min={1}
              value={Number(d.sliceStep ?? 1)}
              onChange={(event) => updateNumberField('sliceStep', event.target.value, 1)}
              style={inputStyle}
            />
          </div>
        )}
      </div>

      {runMode === 'range' ? (
        <div style={rangeGridStyle}>
          <div>
            <label style={labelStyle}>Start</label>
            <input
              type="number"
              min={0}
              value={Number(d.sliceStart ?? 0)}
              onChange={(event) => updateNumberField('sliceStart', event.target.value)}
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>End</label>
            <input
              type="number"
              min={0}
              value={Number(d.sliceEnd ?? 0)}
              onChange={(event) => updateNumberField('sliceEnd', event.target.value)}
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Step</label>
            <input
              type="number"
              min={1}
              value={Number(d.sliceStep ?? 1)}
              onChange={(event) => updateNumberField('sliceStep', event.target.value, 1)}
              style={inputStyle}
            />
          </div>
        </div>
      ) : null}

      {runMode === 'wholeVolume' ? (
        <NodeHint>
          Whole scan mode runs SAM2 slice by slice in the selected view. This can take time.
        </NodeHint>
      ) : null}

      {configError ? (
        <NodeHint>
          {configError}. Using local fallback profiles.
        </NodeHint>
      ) : null}

      {!hasSession ? (
        <NodeHint>
          For manual slice choice, connect Data Loader directly to Interactive Annotator and run SAM2 there.
        </NodeHint>
      ) : null}

      {result ? (
        <div style={summaryStyle}>
          <div style={{ color: 'var(--accent-green)', fontWeight: 800, marginBottom: 4 }}>
            {result.count} polygon{result.count === 1 ? '' : 's'} across {result.segmentedSliceCount || 1} slice{(result.segmentedSliceCount || 1) === 1 ? '' : 's'}
          </div>
          <div style={metricRowStyle}>
            <span>Raw masks</span>
            <strong>{result.rawMaskCount}</strong>
          </div>
          <div style={metricRowStyle}>
            <span>Filtered</span>
            <strong>{result.filteredCount}</strong>
          </div>
          <div style={metricRowStyle}>
            <span>Runtime</span>
            <strong>{result.elapsedSeconds.toFixed(1)}s</strong>
          </div>
        </div>
      ) : null}
    </BaseNode>
  );
}

export default memo(MedSAM2SegmenterNode);
