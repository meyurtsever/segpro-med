import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { ExportNodeData } from '../types/nodes';

const EXPORT_INFO: NodeInfo = {
  description:
    'Exports a stored annotation record to a file path that can continue through Data I/O workflow branches.',
  inputs: ['Annotation record from Annotation Store or Annotation Load', 'Annotated session when a record can be stored first'],
  outputs: ['Exported file path'],
  tips: [
    'JSON export uses the existing AnnotationManager export path.',
    'Run Annotation Store before Export when annotations have not been persisted yet.',
  ],
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 8,
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginTop: 6,
};

function ExportNode({ id, data }: NodeProps) {
  const d = data as unknown as ExportNodeData;

  return (
    <BaseNode
      nodeId={id}
      nodeType="exportNode"
      title="Export"
      icon="EX"
      color="var(--accent-orange)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={EXPORT_INFO}
    >
      <div style={rowStyle}>
        <span>Format</span>
        <strong>{d.exportFormat || 'json'}</strong>
      </div>
      <div style={rowStyle}>
        <span>Annotations</span>
        <strong>{d.annotationCount ?? d.annotationRecord?.annotationCount ?? 0}</strong>
      </div>
      {d.outputPath ? (
        <div style={{ ...rowStyle, display: 'block', wordBreak: 'break-word' }}>
          <span>{d.outputPath}</span>
        </div>
      ) : (
        <NodeHint>
          Connect an annotation record, then run this node to create an export path.
        </NodeHint>
      )}
    </BaseNode>
  );
}

export default memo(ExportNode);
