import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { AnnotationStoreNodeData } from '../types/nodes';

const STORE_INFO: NodeInfo = {
  description:
    'Persists annotations produced by Interactive Annotator or Annotation Load using the backend annotation store.',
  inputs: ['Annotated session from Interactive Annotator or Annotation Load'],
  outputs: ['Annotation record for export or downstream bookkeeping'],
  tips: [
    'Use the same user and study path in Annotation Load to restore saved annotations.',
    'Store runs during workflow execution; canvas edits remain local until the node is run.',
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

function AnnotationStoreNode({ id, data }: NodeProps) {
  const d = data as unknown as AnnotationStoreNodeData;
  const hasStudy = Boolean(d.studyPath?.trim());

  return (
    <BaseNode
      nodeId={id}
      nodeType="annotationStore"
      title="Annotation Store"
      icon="AS"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={STORE_INFO}
    >
      <div style={rowStyle}>
        <span>User</span>
        <strong>{d.userId || 'workflow_user'}</strong>
      </div>
      <div style={rowStyle}>
        <span>Stored</span>
        <strong>{d.annotationCount ?? d.annotationRecord?.annotationCount ?? 0}</strong>
      </div>
      {hasStudy ? (
        <div style={{ ...rowStyle, display: 'block', wordBreak: 'break-word' }}>
          <span>{d.studyPath}</span>
        </div>
      ) : (
        <NodeHint>
          Set a study path or connect an annotator that inherited one from Data Loader.
        </NodeHint>
      )}
    </BaseNode>
  );
}

export default memo(AnnotationStoreNode);
