/**
 * AutoSegmentationNode
 * =====================
 * Configuration node for SAM2 automatic segmentation.
 *
 * Provides a dropdown to select a segmentation profile (fast, balanced,
 * high_detail, tumor_detection, mammography, etc.) and shows profile details.
 *
 * This node acts as a "configuration provider" — it does NOT run segmentation
 * itself. When connected to an InteractiveAnnotator, it enables the
 * "▶ Run Auto Segmentation" button inside the annotator, which calls the
 * backend endpoint with the selected config.
 */

import { memo, useCallback, useEffect, useState } from 'react';
import { type NodeProps } from '@xyflow/react';
import BaseNode from './BaseNode';
import useWorkflowStore from '../store/workflowStore';
import type { AutoSegmentationNodeData } from '../types/nodes';
import * as api from '../api/client';
import type { NodeInfo } from '../components/InfoModal';
import NodeHint from '../components/NodeHint';

// ---------------------------------------------------------------------------
// Info modal content
// ---------------------------------------------------------------------------

const AUTO_SEG_INFO: NodeInfo = {
  description:
    'Configures SAM2 automatic segmentation for medical images. ' +
    'Connect to an Interactive Annotator to enable the "Run Auto Segmentation" button. ' +
    'SAM2 generates region masks which are converted to polygon annotations.',
  inputs: [
    'Data Source — a loaded medical imaging session (from Data Loader or Format Converter)',
  ],
  outputs: [
    'Config → Annotator — provides segmentation config to a connected Interactive Annotator',
  ],
  tips: [
    'Uses SAM2AutomaticMaskGenerator with configurable profiles',
    'Profiles: fast, balanced, high_detail, tumor_detection, mammography, etc.',
    'Masks are automatically converted to polygon annotations',
    'Each polygon is labeled with IoU score and area',
  ],
};

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

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

const statusDotStyle = (connected: boolean): React.CSSProperties => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  background: connected ? 'var(--accent-green)' : 'var(--text-muted)',
  display: 'inline-block',
  marginRight: 6,
  flexShrink: 0,
});

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function AutoSegmentationNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as AutoSegmentationNodeData;

  const [loadingConfigs, setLoadingConfigs] = useState(() => d.availableConfigs.length === 0);
  const [configError, setConfigError] = useState<string | null>(null);

  // Fetch available configs from backend on mount
  useEffect(() => {
    if (d.availableConfigs.length === 0) {
      api.getSegmentationConfigs()
        .then((res) => {
          setConfigError(null);
          updateNodeData(id, {
            availableConfigs: res.configs,
          } as Partial<AutoSegmentationNodeData>);
        })
        .catch((err) => {
          setConfigError(err instanceof Error ? err.message : 'Failed to load segmentation configs');
        })
        .finally(() => setLoadingConfigs(false));
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfigChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { configName: e.target.value } as Partial<AutoSegmentationNodeData>);
    },
    [id, updateNodeData],
  );

  const selectedConfig = d.availableConfigs.find((c) => c.name === d.configName);
  const hasSession = Boolean(d.sessionId);

  return (
    <BaseNode
      nodeId={id}
      nodeType="autoSegmentation"
      title="Auto Segmentation"
      icon="🔬"
      color="var(--accent-purple)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={AUTO_SEG_INFO}
    >
      <div>
        {/* Connection status */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8, fontSize: 11 }}>
          <span style={statusDotStyle(hasSession)} />
          <span style={{ color: hasSession ? 'var(--accent-green)' : 'var(--text-muted)' }}>
            {hasSession ? 'Data connected' : 'Awaiting data connection'}
          </span>
        </div>

        {/* Config selector */}
        <label style={labelStyle}>Segmentation Profile</label>
        <select
          value={d.configName}
          onChange={handleConfigChange}
          style={selectStyle}
          disabled={loadingConfigs}
        >
          {loadingConfigs ? (
            <option>Loading configs...</option>
          ) : d.availableConfigs.length > 0 ? (
            d.availableConfigs.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
              </option>
            ))
          ) : (
            // Fallback if backend not reached
            <>
              <option value="fast">Fast</option>
              <option value="balanced">Balanced</option>
              <option value="high_detail">High Detail</option>
              <option value="small_structures">Small Structures</option>
              <option value="tumor_detection">Tumor Detection</option>
              <option value="skull_stripping">Skull Stripping</option>
              <option value="mammography">Mammography</option>
            </>
          )}
        </select>

        {/* Config details */}
        {selectedConfig && (
          <div style={detailBoxStyle}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
              {selectedConfig.description}
            </div>
            <div>Grid: {selectedConfig.points_per_side}×{selectedConfig.points_per_side} pts</div>
            <div>IoU threshold: {selectedConfig.pred_iou_thresh}</div>
            <div>Min area: {selectedConfig.min_mask_region_area} px</div>
          </div>
        )}

        {configError && (
          <div
            style={{
              marginTop: 8,
              padding: '6px 8px',
              background: 'rgba(224, 92, 92, 0.08)',
              border: '1px solid rgba(224, 92, 92, 0.25)',
              borderRadius: 6,
              color: 'var(--accent-red)',
              fontSize: 10,
              lineHeight: 1.4,
            }}
          >
            {configError}. Using local fallback profiles.
          </div>
        )}

        {/* Hint */}
        <NodeHint>
          <span>
            {hasSession
              ? '→ Connect to Interactive Annotator to run'
              : '← Connect Data Loader first'}
          </span>
        </NodeHint>
      </div>
    </BaseNode>
  );
}

export default memo(AutoSegmentationNode);
