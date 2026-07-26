/**
 * PatientAssignNode
 * =================
 * Assigns campaign patients to an expert using the existing crowdsourcing
 * assignment JSON managed by the Gradio app.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
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

function parsePatientIds(value: string | undefined) {
  return (value || '')
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatPatientIds(values: string[]) {
  return values.join('\n');
}

function PatientAssignNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as PatientAssignNodeData;
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const lastAutoRefreshKey = useRef<string | null>(null);
  const hasCreatedCampaign = Boolean(d.campaign);

  const updateField = useCallback(
    (key: keyof PatientAssignNodeData, value: string) => {
      updateNodeData(id, { [key]: value } as Partial<PatientAssignNodeData>);
    },
    [id, updateNodeData],
  );

  const refreshOptions = useCallback(async (force = false) => {
    if (!d.campaignName?.trim()) return;
    if (force && !hasCreatedCampaign) {
      setLocalError('Run Campaign Setup first to create or reuse this campaign.');
      return;
    }

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
      const message = error instanceof Error ? error.message : 'Assignment options failed to load';
      setLocalError(
        message.includes('Campaign not found')
          ? 'Campaign has not been created yet. Run Campaign Setup first.'
          : message,
      );
    } finally {
      setBusy(false);
    }
  }, [d.campaignName, hasCreatedCampaign, id, updateNodeData]);

  useEffect(() => {
    if (!d.campaignName || !hasCreatedCampaign) return;
    const key = `${d.campaignName}:${d.campaign?.createdAt || ''}:${d.campaign?.progress?.assignedPatients ?? 0}:${d.campaign?.progress?.unassignedPatients ?? 0}`;
    if (lastAutoRefreshKey.current === key) return;
    lastAutoRefreshKey.current = key;
    refreshOptions(false);
  }, [
    d.campaign?.createdAt,
    d.campaign?.progress?.assignedPatients,
    d.campaign?.progress?.unassignedPatients,
    d.campaignName,
    hasCreatedCampaign,
    refreshOptions,
  ]);

  const previewPatients = useMemo(() => d.previewPatients || [], [d.previewPatients]);
  const unassigned = useMemo(() => d.unassignedPatients || [], [d.unassignedPatients]);
  const patientOptions = useMemo(
    () => (hasCreatedCampaign ? unassigned : previewPatients),
    [hasCreatedCampaign, previewPatients, unassigned],
  );
  const experts = useMemo(() => d.availableExperts || [], [d.availableExperts]);
  const preview = patientOptions.slice(0, 4).join(', ');
  const assignedCount = d.campaign?.progress.assignedPatients ?? d.assignmentCount ?? 0;
  const selectedPatientIds = parsePatientIds(d.patientIdsText);

  const setSelectedPatient = useCallback(
    (patientId: string, checked: boolean) => {
      const current = new Set(parsePatientIds(d.patientIdsText));
      if (checked) {
        current.add(patientId);
      } else {
        current.delete(patientId);
      }
      updateNodeData(id, {
        patientIdsText: formatPatientIds(
          patientOptions.filter((candidate) => current.has(candidate)),
        ),
      } as Partial<PatientAssignNodeData>);
    },
    [d.patientIdsText, id, patientOptions, updateNodeData],
  );

  const setAllSelectedPatients = useCallback(
    (checked: boolean) => {
      updateNodeData(id, {
        patientIdsText: checked ? formatPatientIds(patientOptions) : '',
      } as Partial<PatientAssignNodeData>);
    },
    [id, patientOptions, updateNodeData],
  );

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
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        <div style={{
          border: '1px solid color-mix(in srgb, var(--accent-green) 28%, var(--border-color))',
          borderRadius: 7,
          padding: '8px 9px',
          background: 'color-mix(in srgb, var(--accent-green) 7%, var(--bg-secondary))',
        }}>
          <div style={{ color: 'var(--accent-green)', fontSize: 20, fontWeight: 900 }}>
            {patientOptions.length}
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            {hasCreatedCampaign ? 'Unassigned' : 'Patients'}
          </div>
        </div>
        <div style={{
          border: '1px solid color-mix(in srgb, var(--accent-blue) 28%, var(--border-color))',
          borderRadius: 7,
          padding: '8px 9px',
          background: 'color-mix(in srgb, var(--accent-blue) 7%, var(--bg-secondary))',
        }}>
          <div style={{ color: 'var(--accent-blue)', fontSize: 20, fontWeight: 900 }}>
            {assignedCount}
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Assigned</div>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setDetailsOpen((value) => !value)}
        style={{ ...buttonStyle(false), marginTop: 8, width: '100%' }}
      >
        {detailsOpen ? 'Collapse Assignment' : 'Expand Assignment'}
      </button>

      {detailsOpen ? (
        <>
      <label style={labelStyle}>Campaign</label>
      <div style={{
        padding: '7px 8px',
        borderRadius: 5,
        border: '1px solid var(--border-color)',
        background: 'rgba(79, 141, 245, 0.08)',
        color: d.campaignName ? 'var(--text-primary)' : 'var(--text-muted)',
        fontSize: 12,
        fontWeight: 800,
      }}>
        {d.campaignName || 'Connect Campaign Setup and run it first'}
      </div>
      {!hasCreatedCampaign && d.campaignName ? (
        <NodeHint>
          "{d.campaignName}" is only a planned campaign name. Run Campaign Setup to create or reuse it before assigning patients.
        </NodeHint>
      ) : null}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 96px', gap: 6, marginTop: 8 }}>
        <div>
          <label style={labelStyle}>Expert</label>
          <select
            value={d.expertId || ''}
            onChange={(event) => updateField('expertId', event.target.value)}
            style={selectStyle}
            disabled={!hasCreatedCampaign || experts.length === 0}
          >
            <option value="">
              {!hasCreatedCampaign ? 'Run setup first' : experts.length ? 'Select expert' : 'No experts found'}
            </option>
            {experts.map((expert) => (
              <option key={expert} value={expert}>{expert}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={labelStyle}>Mode</label>
          <select
            value={d.assignmentMode || 'allUnassigned'}
            onChange={(event) => updateField('assignmentMode', event.target.value)}
            style={inputStyle}
            disabled={!hasCreatedCampaign}
          >
            <option value="allUnassigned">All</option>
            <option value="selected">Selected</option>
          </select>
        </div>
      </div>

      {d.assignmentMode === 'selected' ? (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <label style={{ ...labelStyle, marginBottom: 0 }}>Patients</label>
            <button
              type="button"
              onClick={() => setAllSelectedPatients(selectedPatientIds.length !== unassigned.length)}
              style={{ ...buttonStyle(false), padding: '4px 6px' }}
              disabled={!hasCreatedCampaign || patientOptions.length === 0}
            >
              {selectedPatientIds.length === patientOptions.length && patientOptions.length > 0 ? 'Clear' : 'Select All'}
            </button>
          </div>
          <div style={{
            maxHeight: 132,
            overflowY: 'auto',
            border: '1px solid var(--border-color)',
            borderRadius: 6,
            background: 'var(--bg-tertiary)',
            padding: 6,
          }}>
            {patientOptions.length > 0 ? patientOptions.map((patientId) => {
              const checked = selectedPatientIds.includes(patientId);
              return (
                <label
                  key={patientId}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 7,
                    padding: '5px 6px',
                    borderRadius: 5,
                    color: checked ? 'var(--accent-green)' : 'var(--text-secondary)',
                    fontSize: 12,
                    fontWeight: 800,
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!hasCreatedCampaign}
                    onChange={(event) => setSelectedPatient(patientId, event.target.checked)}
                  />
                  <span>{patientId}</span>
                </label>
              );
            }) : (
              <div style={{ color: 'var(--text-muted)', fontSize: 11, padding: 6 }}>
                {hasCreatedCampaign ? 'No unassigned patients available.' : 'No scanned patients available.'}
              </div>
            )}
          </div>
          <div style={{ marginTop: 4, color: 'var(--text-muted)', fontSize: 10 }}>
          {selectedPatientIds.length} selected
          </div>
        </div>
      ) : null}

      {d.assignmentMode !== 'selected' && patientOptions.length > 0 ? (
        <div style={{ marginTop: 8 }}>
          <label style={labelStyle}>{hasCreatedCampaign ? 'Unassigned Patients' : 'Scanned Patients'}</label>
          <div style={{
            maxHeight: 92,
            overflowY: 'auto',
            border: '1px solid var(--border-color)',
            borderRadius: 6,
            background: 'var(--bg-tertiary)',
            padding: 6,
          }}>
            {patientOptions.map((patientId) => (
              <div
                key={patientId}
                style={{
                  padding: '5px 6px',
                  borderBottom: '1px solid rgba(255,255,255,0.05)',
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  fontWeight: 800,
                }}
              >
                {patientId}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 8 }}>
        <button type="button" onClick={() => refreshOptions(true)} style={buttonStyle(busy)} disabled={busy || !d.campaignName}>
          Refresh
        </button>
        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
          {hasCreatedCampaign
            ? `${unassigned.length} unassigned, ${experts.length} experts`
            : `${previewPatients.length} scanned, run setup first`}
        </span>
      </div>

      {!hasCreatedCampaign ? (
        <NodeHint>
          Run Campaign Setup first. Then this node will load experts and make patients selectable for assignment.
        </NodeHint>
      ) : !d.expertId ? (
        <NodeHint>
          Select an expert, then run this node to assign the current unassigned patient set.
        </NodeHint>
      ) : d.assignmentCount !== undefined ? (
        <div style={{ marginTop: 8, color: 'var(--accent-green)', fontSize: 11, fontWeight: 800 }}>
          Assigned {d.assignmentCount} patient{d.assignmentCount === 1 ? '' : 's'} to {d.expertId || 'expert'}
        </div>
      ) : unassigned.length > 0 ? (
        <NodeHint>
          Next unassigned patients: {preview}{unassigned.length > 4 ? ` and ${unassigned.length - 4} more` : ''}. Run this node to assign them to {d.expertId || 'the selected expert'}.
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
        </>
      ) : null}
    </BaseNode>
  );
}

export default memo(PatientAssignNode);
