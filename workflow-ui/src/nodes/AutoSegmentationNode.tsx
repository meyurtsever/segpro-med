/**
 * SegmentationProfileNode
 * =======================
 * Configuration-only node for SAM2 profile selection.
 *
 * It does not run the model. Interactive Annotator runs SAM2 on the visible
 * slice, while Batch SAM2 Segmenter uses this profile for automation.
 */

import { memo, useCallback, useEffect, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { AutoSegmentationNodeData } from '../types/nodes';
import * as api from '../api/client';

const PROFILE_INFO: NodeInfo = {
  description:
    'Stores the SAM2 segmentation profile used by Interactive Annotator and Batch SAM2 Segmenter. This node is configuration only.',
  inputs: [
    'No data input required.',
  ],
  outputs: [
    'Segmentation profile for visible-slice annotation or batch SAM2 automation.',
  ],
  tips: [
    'Connect to Interactive Annotator when users should run SAM2 on the slice they are viewing.',
    'Connect to Batch SAM2 Segmenter when SAM2 should run without manual slice selection.',
    'If no profile is connected to Annotator, it uses the fast profile by default.',
  ],
};

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
  fontWeight: 500,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
};

const selectStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-primary)',
  fontSize: 12,
  outline: 'none',
  cursor: 'pointer',
};

const detailBoxStyle: React.CSSProperties = {
  marginTop: 8,
  padding: '8px 10px',
  background: 'var(--bg-tertiary)',
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  fontSize: 11,
  color: 'var(--text-secondary)',
  lineHeight: 1.5,
};

const statusDotStyle: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: '50%',
  background: 'var(--accent-green)',
  display: 'inline-block',
  marginRight: 6,
  flexShrink: 0,
};

function titleCase(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function AutoSegmentationNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as AutoSegmentationNodeData;
  const [loadingConfigs, setLoadingConfigs] = useState(() => d.availableConfigs.length === 0);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    if (d.availableConfigs.length > 0) return;

    api.getSegmentationConfigs()
      .then((res) => {
        setConfigError(null);
        updateNodeData(id, {
          availableConfigs: res.configs,
        } as Partial<AutoSegmentationNodeData>);
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
      updateNodeData(id, { configName: event.target.value } as Partial<AutoSegmentationNodeData>);
    },
    [id, updateNodeData],
  );

  const selectedConfig = d.availableConfigs.find((config) => config.name === d.configName);
  const configOptions = d.availableConfigs.length > 0
    ? d.availableConfigs.map((config) => config.name)
    : FALLBACK_CONFIGS;

  return (
    <BaseNode
      nodeId={id}
      nodeType="autoSegmentation"
      title="Segmentation Profile"
      icon="CFG"
      color="var(--accent-purple)"
      status={d.status}
      error={d.error}
      hasInput={false}
      hasOutput={true}
      info={PROFILE_INFO}
    >
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8, fontSize: 11 }}>
        <span style={statusDotStyle} />
        <span style={{ color: 'var(--accent-green)' }}>
          Profile ready
        </span>
      </div>

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

      {selectedConfig ? (
        <div style={detailBoxStyle}>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            {selectedConfig.description}
          </div>
          <div>Grid: {selectedConfig.points_per_side} x {selectedConfig.points_per_side} pts</div>
          <div>IoU threshold: {selectedConfig.pred_iou_thresh}</div>
          <div>Min area: {selectedConfig.min_mask_region_area} px</div>
        </div>
      ) : null}

      {configError ? (
        <NodeHint
          style={{
            color: 'var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.08)',
            borderColor: 'rgba(224, 92, 92, 0.25)',
          }}
        >
          {configError}. Using local fallback profiles.
        </NodeHint>
      ) : null}

      <NodeHint>
        Connect to Interactive Annotator for visible-slice SAM2, or Batch SAM2 Segmenter for automation.
      </NodeHint>
    </BaseNode>
  );
}

export default memo(AutoSegmentationNode);
