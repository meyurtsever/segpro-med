/**
 * CrowdsourcingTasksNode
 * ======================
 * Assignment guide shown when an expert enters crowdsourcing mode.
 * It replaces campaign-level status for task execution so the expert sees the
 * current patient and the remaining queue directly on the canvas.
 */

import { memo, useCallback, useState, type CSSProperties } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import type { CrowdsourcingTaskItem, CrowdsourcingTasksNodeData } from '../types/nodes';

const TASKS_INFO: NodeInfo = {
  description:
    'Shows the expert assignment queue for crowdsourcing annotation. Use it to confirm the current patient, remaining tasks, and completed tasks while annotating.',
  inputs: [],
  outputs: [],
  tips: [
    'Annotate the current patient in Interactive Annotator.',
    'Use Submit Task only after the current patient has the needed annotations.',
    'Use Next Task to skip to another pending assignment without submitting.',
  ],
};

const tableHeaderStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1.05fr 1fr 0.7fr 58px',
  gap: 6,
  padding: '7px 8px',
  borderBottom: '1px solid var(--border-color)',
  color: 'var(--text-muted)',
  fontSize: 10,
  fontWeight: 900,
  textTransform: 'uppercase',
};

const taskRowStyle = (task: CrowdsourcingTaskItem): CSSProperties => {
  const color = task.status === 'current'
    ? 'var(--accent-green)'
    : task.status === 'completed'
      ? 'var(--accent-blue)'
      : 'var(--text-secondary)';

  return {
    display: 'grid',
    gridTemplateColumns: '1.05fr 1fr 0.7fr 58px',
    gap: 6,
    padding: '8px',
    borderBottom: '1px solid rgba(255,255,255,0.06)',
    background: task.status === 'current'
      ? 'rgba(76, 175, 139, 0.1)'
      : 'transparent',
    color,
    fontSize: 12,
    alignItems: 'center',
    cursor: task.status === 'current' ? 'default' : 'pointer',
  };
};

const smallButtonStyle: CSSProperties = {
  border: '1px solid color-mix(in srgb, var(--accent-green) 42%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-green) 12%, var(--bg-secondary))',
  color: 'var(--accent-green)',
  borderRadius: 6,
  padding: '5px 8px',
  fontSize: 11,
  fontWeight: 900,
  cursor: 'pointer',
};

function statusLabel(status: CrowdsourcingTaskItem['status']) {
  if (status === 'current') return 'Current';
  if (status === 'completed') return 'Done';
  return 'Pending';
}

function CrowdsourcingTasksNode({ id, data }: NodeProps) {
  const d = data as unknown as CrowdsourcingTasksNodeData;
  const [showAllTasks, setShowAllTasks] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const tasks = d.tasks || [];
  const currentTask = tasks.find((task) => task.status === 'current');
  const visibleTasks = showAllTasks ? tasks : tasks.filter((task) => task.status !== 'completed');
  const pendingCount = tasks.filter((task) => task.status === 'pending' || task.status === 'current').length;
  const completedCount = tasks.filter((task) => task.status === 'completed').length;
  const handleSelectTask = useCallback(
    (task: CrowdsourcingTaskItem) => {
      if (task.status === 'current') return;
      if (!d.onSelectTask) return;
      d.onSelectTask(task.campaignId, task.patientId);
    },
    [d],
  );

  return (
    <BaseNode
      nodeId={id}
      nodeType="crowdsourcingTasks"
      title="Your Assignment Tasks"
      icon="AT"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={false}
      hasOutput={false}
      info={TASKS_INFO}
    >
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 8,
        marginBottom: 9,
      }}>
        <div style={{
          border: '1px solid rgba(76, 175, 139, 0.28)',
          borderRadius: 7,
          padding: '10px 11px',
          background: 'rgba(76, 175, 139, 0.08)',
        }}>
          <div style={{ color: 'var(--accent-green)', fontSize: 26, fontWeight: 900, lineHeight: 1 }}>
            {pendingCount}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, fontWeight: 800, marginTop: 5 }}>
            Pending
          </div>
        </div>
        <div style={{
          border: '1px solid rgba(79, 141, 245, 0.26)',
          borderRadius: 7,
          padding: '10px 11px',
          background: 'rgba(79, 141, 245, 0.07)',
        }}>
          <div style={{ color: 'var(--accent-blue)', fontSize: 26, fontWeight: 900, lineHeight: 1 }}>
            {completedCount}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, fontWeight: 800, marginTop: 5 }}>
            Done
          </div>
        </div>
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 8,
        alignItems: 'center',
      }}>
        <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900, minWidth: 0 }}>
          {currentTask ? `Current: ${currentTask.patientId}` : 'No current assignment'}
        </div>
        <button
          type="button"
          onClick={() => setDetailsOpen((value) => !value)}
          style={smallButtonStyle}
          title={detailsOpen ? 'Hide assignment details' : 'Show assignment details'}
        >
          {detailsOpen ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {detailsOpen ? (
        <>
      {currentTask ? (
        <div style={{
          border: '1px solid rgba(76, 175, 139, 0.28)',
          borderRadius: 6,
          padding: '7px 8px',
          background: 'rgba(76, 175, 139, 0.08)',
          marginTop: 9,
          marginBottom: 8,
        }}>
          <div style={{ color: 'var(--text-muted)', fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>
            Current Patient
          </div>
          <div style={{ color: 'var(--text-primary)', fontSize: 15, fontWeight: 900, marginTop: 2 }}>
            {currentTask.patientId}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 2 }}>
            {currentTask.campaignId}{currentTask.modality ? ` - ${currentTask.modality.toUpperCase()}` : ''}
          </div>
        </div>
      ) : null}

      {tasks.length > 0 ? (
        <>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 8,
          marginBottom: 6,
        }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: 12, fontWeight: 800 }}>
            {showAllTasks ? 'Showing all assignments' : 'Showing current and pending'}
          </div>
          <button
            type="button"
            onClick={() => setShowAllTasks((value) => !value)}
            style={smallButtonStyle}
          >
            {showAllTasks ? 'Hide Done' : 'Show All'}
          </button>
        </div>
        <div style={{
          border: '1px solid var(--border-color)',
          borderRadius: 6,
          overflow: 'hidden',
          maxHeight: 230,
          overflowY: 'auto',
        }}>
          <div style={tableHeaderStyle}>
            <span>Patient</span>
            <span>Campaign</span>
            <span>Status</span>
            <span>Action</span>
          </div>
          {visibleTasks.map((task) => (
            <div
              key={`${task.campaignId}:${task.patientId}`}
              style={taskRowStyle(task)}
              role="button"
              tabIndex={task.status === 'current' ? -1 : 0}
              title={task.status === 'current' ? 'Current assignment' : `Load ${task.patientId}`}
              onClick={(event) => {
                event.stopPropagation();
                handleSelectTask(task);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  event.stopPropagation();
                  handleSelectTask(task);
                }
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {task.patientId}
              </span>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {task.campaignId}
              </span>
              <span style={{ fontWeight: 900 }}>{statusLabel(task.status)}</span>
              <button
                type="button"
                disabled={task.status === 'current' || !d.onSelectTask}
                onClick={(event) => {
                  event.stopPropagation();
                  handleSelectTask(task);
                }}
                style={{
                  ...smallButtonStyle,
                  padding: '4px 6px',
                  fontSize: 10,
                  opacity: task.status === 'current' || !d.onSelectTask ? 0.55 : 1,
                  cursor: task.status === 'current' || !d.onSelectTask ? 'default' : 'pointer',
                }}
              >
                {task.status === 'current' ? 'Open' : 'Load'}
              </button>
            </div>
          ))}
        </div>
        </>
      ) : (
        <NodeHint>No assigned tasks were found for this expert.</NodeHint>
      )}
      </>
      ) : null}
    </BaseNode>
  );
}

export default memo(CrowdsourcingTasksNode);
