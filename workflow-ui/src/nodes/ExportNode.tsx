import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { ExportNodeData } from '../types/nodes';

const EXPORT_INFO: NodeInfo = {
  description:
    'Marks a workflow output as ready for download, transfer, or downstream use. It supports annotation records and deidentified data outputs.',
  inputs: ['Annotation record from Annotation Store or Annotation Load', 'Annotated session when a record can be stored first', 'Deidentified file path from Deidentify'],
  outputs: ['Exported file path'],
  tips: [
    'JSON export uses the existing AnnotationManager export path.',
    'For deidentified data, connect Deidentify directly and the sanitized output path becomes the exported artifact.',
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
  const isDataExport = Boolean(d.outputPath) && !d.annotationRecord;
  const nodeColor = isDataExport ? 'var(--accent-green)' : 'var(--accent-orange)';

  return (
    <BaseNode
      nodeId={id}
      nodeType="exportNode"
      title={d.label || 'Export'}
      icon="EX"
      color={nodeColor}
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={EXPORT_INFO}
    >
      <div style={rowStyle}>
        <span>Format</span>
        <strong>{isDataExport ? (d.fileType || 'data') : (d.exportFormat || 'json')}</strong>
      </div>
      <div style={rowStyle}>
        <span>{isDataExport ? 'Output' : 'Annotations'}</span>
        <strong>{isDataExport ? 'ready' : (d.annotationCount ?? d.annotationRecord?.annotationCount ?? 0)}</strong>
      </div>
      {d.outputPath ? (
        <div style={{ ...rowStyle, display: 'block', wordBreak: 'break-word' }}>
          <div style={{
            color: nodeColor,
            fontSize: 11,
            fontWeight: 900,
            marginBottom: 5,
          }}>
            Export completed successfully.
          </div>
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
