/**
 * PatientAssignNode
 * =================
 * Assigns campaign patients to an expert using the existing crowdsourcing
 * assignment JSON managed by the Gradio app.
 */

import { memo, useCallback, useEffect, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { PatientAssignNodeData } from '../types/nodes';
import * as api from '../api/client';

const ASSIGN_INFO: NodeInfo = {
  description:
    'Assigns campaign patients to one expert annotator. Use multiple assignment nodes when the campaign should be split across multiple experts.',
  inputs: ['Campaign from Campaign Setup'],
  outputs: ['Assignment result with expert, patient IDs, and refreshed campaign status'],
  tips: [
    'All unassigned is the safest bulk mode for first-pass distribution.',
    'Selected IDs mode is useful when an admin wants a specific subset for a specific expert.',
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

const buttonStyle = (active = false): React.CSSProperties => ({
  padding: '6px 8px',
  borderRadius: 5,
  border: `1px solid ${active ? 'var(--accent-green)' : 'var(--border-color)'}`,
  background: active ? 'rgba(76, 175, 139, 0.14)' : 'var(--bg-tertiary)',
  color: active ? 'var(--accent-green)' : 'var(--text-secondary)',
  fontSize: 10,
  fontWeight: 800,
  cursor: active ? 'progress' : 'pointer',
});

function PatientAssignNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as PatientAssignNodeData;
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const updateField = useCallback(
    (key: keyof PatientAssignNodeData, value: string) => {
      updateNodeData(id, { [key]: value } as Partial<PatientAssignNodeData>);
    },
    [id, updateNodeData],
  );

  const refreshOptions = useCallback(async () => {
    if (!d.campaignName?.trim()) return;

    setBusy(true);
    setLocalError(null);
    try {
      const res = await api.getCollaborationAssignmentOptions(d.campaignName);
      updateNodeData(id, {
        availableExperts: res.experts,
        unassignedPatients: res.unassigned_patients,
        campaign: {
          name: res.campaign.name,
          datasetPath: res.campaign.dataset_path,
          description: res.campaign.description,
          createdAt: res.campaign.created_at,
          totalPatients: res.campaign.total_patients,
          patients: res.campaign.patients,
          progress: {
            totalPatients: res.campaign.progress.total_patients,
            assignedPatients: res.campaign.progress.assigned_patients,
            completed: res.campaign.progress.completed,
            reviewed: res.campaign.progress.reviewed,
            unassignedPatients: res.campaign.progress.unassigned_patients,
          },
          unassignedPatients: res.campaign.unassigned_patients,
          assignments: res.campaign.assignments.map((assignment) => ({
            expertId: assignment.expert_id,
            assignedPatients: assignment.assigned_patients,
            completedPatients: assignment.completed_patients,
            pendingPatients: assignment.pending_patients,
          })),
        },
      } as Partial<PatientAssignNodeData>);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Assignment options failed to load');
    } finally {
      setBusy(false);
    }
  }, [d.campaignName, id, updateNodeData]);

  useEffect(() => {
    refreshOptions();
  }, [refreshOptions]);

  const unassigned = d.unassignedPatients || [];
  const experts = d.availableExperts || [];
  const preview = unassigned.slice(0, 4).join(', ');

  return (
    <BaseNode
      nodeId={id}
      nodeType="patientAssign"
      title="Patient Assign"
      icon="PA"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={ASSIGN_INFO}
    >
      <label style={labelStyle}>Campaign</label>
      <input
        value={d.campaignName || ''}
        onChange={(event) => updateField('campaignName', event.target.value)}
        placeholder="Connect Campaign Setup or type campaign name"
        style={inputStyle}
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 96px', gap: 6, marginTop: 8 }}>
        <div>
          <label style={labelStyle}>Expert ID</label>
          <input
            value={d.expertId || ''}
            onChange={(event) => updateField('expertId', event.target.value)}
            placeholder={experts[0] || 'expert_user'}
            list={`${id}-experts`}
            style={inputStyle}
          />
          <datalist id={`${id}-experts`}>
            {experts.map((expert) => (
              <option key={expert} value={expert} />
            ))}
          </datalist>
        </div>
        <div>
          <label style={labelStyle}>Mode</label>
          <select
            value={d.assignmentMode || 'allUnassigned'}
            onChange={(event) => updateField('assignmentMode', event.target.value)}
            style={inputStyle}
          >
            <option value="allUnassigned">All</option>
            <option value="selected">Selected</option>
          </select>
        </div>
      </div>

      {d.assignmentMode === 'selected' ? (
        <div style={{ marginTop: 8 }}>
          <label style={labelStyle}>Patient IDs</label>
          <textarea
            value={d.patientIdsText || ''}
            onChange={(event) => updateField('patientIdsText', event.target.value)}
            placeholder="patient_001, patient_002"
            rows={3}
            style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.35 }}
          />
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 8 }}>
        <button type="button" onClick={refreshOptions} style={buttonStyle(busy)} disabled={busy || !d.campaignName}>
          Refresh
        </button>
        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
          {unassigned.length} unassigned, {experts.length} experts
        </span>
      </div>

      {d.assignmentCount !== undefined ? (
        <div style={{ marginTop: 8, color: 'var(--accent-green)', fontSize: 11, fontWeight: 800 }}>
          Assigned {d.assignmentCount} patient{d.assignmentCount === 1 ? '' : 's'} to {d.expertId || 'expert'}
        </div>
      ) : unassigned.length > 0 ? (
        <NodeHint>
          Next unassigned patients: {preview}{unassigned.length > 4 ? ` and ${unassigned.length - 4} more` : ''}.
        </NodeHint>
      ) : (
        <NodeHint>
          Connect a campaign and refresh options before running assignment.
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

export default memo(PatientAssignNode);
