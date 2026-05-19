import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { AnnotationLoadNodeData } from '../types/nodes';

const LOAD_INFO: NodeInfo = {
  description:
    'Loads previously persisted annotations for a user and study, then emits an annotated session for review or export.',
  inputs: ['Optional session or file path to inherit study identity'],
  outputs: ['Annotated session', 'Annotation record'],
  tips: [
    'Connect Annotation Load to Interactive Annotator to review and edit saved annotations.',
    'A Data Loader connection can provide the study path automatically after it has run.',
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

function AnnotationLoadNode({ id, data }: NodeProps) {
  const d = data as unknown as AnnotationLoadNodeData;
  const hasStudy = Boolean(d.studyPath?.trim());

  return (
    <BaseNode
      nodeId={id}
      nodeType="annotationLoad"
      title="Annotation Load"
      icon="AL"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={LOAD_INFO}
    >
      <div style={rowStyle}>
        <span>User</span>
        <strong>{d.userId || 'workflow_user'}</strong>
      </div>
      <div style={rowStyle}>
        <span>Loaded</span>
        <strong>{d.annotationCount ?? d.annotations?.length ?? 0}</strong>
      </div>
      {hasStudy ? (
        <div style={{ ...rowStyle, display: 'block', wordBreak: 'break-word' }}>
          <span>{d.studyPath}</span>
        </div>
      ) : (
        <NodeHint>
          Provide a study path before running, or connect a data source that can provide one.
        </NodeHint>
      )}
    </BaseNode>
  );
}

export default memo(AnnotationLoadNode);
