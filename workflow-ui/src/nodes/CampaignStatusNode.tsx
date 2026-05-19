/**
 * CampaignStatusNode
 * ==================
 * Terminal collaboration node that shows campaign progress from the existing
 * crowdsourcing assignment store.
 */

import { memo, useCallback, useEffect, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { CampaignInfo, CampaignStatusNodeData } from '../types/nodes';
import * as api from '../api/client';

const STATUS_INFO: NodeInfo = {
  description:
    'Displays campaign totals, assignment progress, completed submissions, and reviewed annotations.',
  inputs: ['Campaign from Campaign Setup or assignment result from Patient Assign'],
  outputs: [],
  tips: [
    'Use this at the end of a collaboration branch to confirm that assignments changed as expected.',
    'Reviewed counts come from the Gradio Management tab approval flow.',
  ],
};

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
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

const statCardStyle: React.CSSProperties = {
  border: '1px solid rgba(79, 141, 245, 0.24)',
  borderRadius: 6,
  padding: '6px 7px',
  background: 'rgba(79, 141, 245, 0.06)',
};

function mapCampaign(campaign: api.CollaborationCampaign): CampaignInfo {
  return {
    name: campaign.name,
    datasetPath: campaign.dataset_path,
    description: campaign.description,
    createdAt: campaign.created_at,
    totalPatients: campaign.total_patients,
    patients: campaign.patients,
    progress: {
      totalPatients: campaign.progress.total_patients,
      assignedPatients: campaign.progress.assigned_patients,
      completed: campaign.progress.completed,
      reviewed: campaign.progress.reviewed,
      unassignedPatients: campaign.progress.unassigned_patients,
    },
    unassignedPatients: campaign.unassigned_patients,
    assignments: campaign.assignments.map((assignment) => ({
      expertId: assignment.expert_id,
      assignedPatients: assignment.assigned_patients,
      completedPatients: assignment.completed_patients,
      pendingPatients: assignment.pending_patients,
    })),
  };
}

function CampaignStatusNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as CampaignStatusNodeData;
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    if (!d.campaignName?.trim()) return;

    setBusy(true);
    setLocalError(null);
    try {
      const res = await api.getCollaborationCampaign(d.campaignName);
      updateNodeData(id, { campaign: mapCampaign(res) } as Partial<CampaignStatusNodeData>);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Campaign status failed to load');
    } finally {
      setBusy(false);
    }
  }, [d.campaignName, id, updateNodeData]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const campaign = d.campaign;
  const progress = campaign?.progress;
  const completionPercent = progress
    ? Math.round((progress.completed / Math.max(progress.totalPatients, 1)) * 100)
    : 0;

  return (
    <BaseNode
      nodeId={id}
      nodeType="campaignStatus"
      title="Campaign Status"
      icon="ST"
      color="var(--accent-blue)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={false}
      info={STATUS_INFO}
    >
      <label style={labelStyle}>Campaign</label>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          value={d.campaignName || ''}
          onChange={(event) => updateNodeData(id, { campaignName: event.target.value })}
          placeholder="Connect Campaign Setup or Patient Assign"
          style={inputStyle}
        />
        <button
          type="button"
          onClick={refreshStatus}
          disabled={busy || !d.campaignName}
          style={{
            padding: '6px 8px',
            borderRadius: 5,
            border: `1px solid ${busy ? 'var(--accent-blue)' : 'var(--border-color)'}`,
            background: busy ? 'rgba(79, 141, 245, 0.16)' : 'var(--bg-tertiary)',
            color: busy ? 'var(--accent-blue)' : 'var(--text-secondary)',
            fontSize: 10,
            fontWeight: 800,
            cursor: busy ? 'progress' : 'pointer',
          }}
        >
          Refresh
        </button>
      </div>

      {progress ? (
        <>
          <div style={{ marginTop: 9, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <div style={statCardStyle}>
              <div style={{ color: 'var(--accent-blue)', fontSize: 14, fontWeight: 900 }}>
                {progress.totalPatients}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>Total</div>
            </div>
            <div style={statCardStyle}>
              <div style={{ color: 'var(--accent-green)', fontSize: 14, fontWeight: 900 }}>
                {progress.assignedPatients}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>Assigned</div>
            </div>
            <div style={statCardStyle}>
              <div style={{ color: 'var(--accent-orange)', fontSize: 14, fontWeight: 900 }}>
                {progress.completed}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>Completed</div>
            </div>
            <div style={statCardStyle}>
              <div style={{ color: 'var(--accent-purple)', fontSize: 14, fontWeight: 900 }}>
                {progress.reviewed}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>Reviewed</div>
            </div>
          </div>

          <div style={{ marginTop: 9 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
              <span>Completion</span>
              <span>{completionPercent}%</span>
            </div>
            <div style={{ height: 7, borderRadius: 999, background: 'var(--bg-tertiary)', overflow: 'hidden', marginTop: 4 }}>
              <div
                style={{
                  width: `${completionPercent}%`,
                  height: '100%',
                  background: 'var(--accent-green)',
                }}
              />
            </div>
          </div>
        </>
      ) : (
        <NodeHint>
          Connect a campaign branch, then run or refresh to inspect collaboration progress.
        </NodeHint>
      )}

      {localError ? (
        <NodeHint
          style={{
            color: 'var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.08)',
            borderColor: 'rgba(224, 92, 92, 0.25)',
          }}
        >
          {localError}
        </NodeHint>
      ) : null}
    </BaseNode>
  );
}

export default memo(CampaignStatusNode);
