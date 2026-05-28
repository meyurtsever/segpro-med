/**
 * DeidentifyNode
 * ==============
 * Automated PHI removal node for research/export workflows. It keeps the user
 * surface compact while showing the operational evidence experts need: source,
 * destination, sanitized field count, processed file count, and audit location.
 */

import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { DeidentifyNodeData } from '../types/nodes';

const DEIDENTIFY_INFO: NodeInfo = {
  description:
    'Creates a deidentified copy of a loaded study. DICOM PHI tags are blanked, 3D volume data can be defaced, and every operation writes to the anonymization audit trail.',
  inputs: [
    'Loaded session from Data Loader',
    'DICOM directory, DICOM file, NIfTI file, or uploaded ZIP extracted by Data Loader',
  ],
  outputs: [
    'Deidentified file path',
    'Loaded session for the sanitized output',
    'Audit trail entry in db/anonymization_audit.jsonl',
  ],
  tips: [
    'Use Metadata Viewer before and after this node to verify PHI fields were removed.',
    'Keep defacing enabled for 3D head/face volumes before public release or transfer.',
    'The source data is not modified; the node writes a new deidentified output path.',
  ],
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 10,
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginTop: 7,
};

const strongStyle: React.CSSProperties = {
  color: 'var(--text-primary)',
  textAlign: 'right',
};

const pathStyle: React.CSSProperties = {
  marginTop: 8,
  padding: '6px 7px',
  borderRadius: 5,
  border: '1px solid rgba(76, 175, 139, 0.26)',
  background: 'rgba(76, 175, 139, 0.08)',
  color: 'var(--accent-green)',
  fontSize: 10,
  fontFamily: 'monospace',
  lineHeight: 1.35,
  wordBreak: 'break-all',
};

function DeidentifyNode({ id, data }: NodeProps) {
  const d = data as unknown as DeidentifyNodeData;
  const hasOutput = Boolean(d.outputPath);

  return (
    <BaseNode
      nodeId={id}
      nodeType="deidentifyNode"
      title={d.label || 'Deidentify'}
      icon="ID"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={DEIDENTIFY_INFO}
    >
      <div style={rowStyle}>
        <span>Mode</span>
        <strong style={strongStyle}>{d.applyDeface === false ? 'Metadata only' : 'PHI + deface'}</strong>
      </div>
      <div style={rowStyle}>
        <span>PHI fields</span>
        <strong style={strongStyle}>{d.sanitizedFieldCount ?? 0}</strong>
      </div>
      <div style={rowStyle}>
        <span>Files</span>
        <strong style={strongStyle}>{d.filesProcessed ?? 0}</strong>
      </div>
      <div style={rowStyle}>
        <span>Audit</span>
        <strong style={strongStyle}>{d.auditPath ? 'written' : 'pending'}</strong>
      </div>

      {hasOutput ? (
        <div style={pathStyle}>{d.outputPath}</div>
      ) : (
        <NodeHint>
          Connect Data Loader, then run this node to create a deidentified copy.
        </NodeHint>
      )}
    </BaseNode>
  );
}

export default memo(DeidentifyNode);
