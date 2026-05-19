/**
 * CampaignSetupNode
 * =================
 * Creates or reuses a crowdsourcing campaign from the existing Gradio
 * CrowdsourcingManager dataset scan rules.
 */

import { memo, useCallback, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { CampaignSetupNodeData } from '../types/nodes';
import * as api from '../api/client';

const SETUP_INFO: NodeInfo = {
  description:
    'Scans a dataset root for patient folders with valid modalities and creates a crowdsourcing campaign using the same manager as the Gradio Management tab.',
  inputs: [],
  outputs: ['Campaign object with dataset path, patient list, progress, and assignment pool'],
  tips: [
    'Repeated workflow runs reuse an existing campaign name instead of overwriting assignments.',
    'Use one Patient Assign node per expert when distributing a campaign across several annotators.',
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
  border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border-color)'}`,
  background: active ? 'rgba(79, 141, 245, 0.16)' : 'var(--bg-tertiary)',
  color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
  fontSize: 10,
  fontWeight: 800,
  cursor: active ? 'progress' : 'pointer',
});

const statStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: 6,
  marginTop: 8,
};

function CampaignSetupNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as CampaignSetupNodeData;
  const [busy, setBusy] = useState<'scan' | 'pick' | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const updateField = useCallback(
    (key: keyof CampaignSetupNodeData, value: string) => {
      updateNodeData(id, { [key]: value } as Partial<CampaignSetupNodeData>);
    },
    [id, updateNodeData],
  );

  const pickDirectory = useCallback(async () => {
    setBusy('pick');
    setLocalError(null);
    try {
      const res = await api.openNativePathDialog('directory', d.datasetPath || '');
      if (!res.cancelled && res.path) {
        updateNodeData(id, { datasetPath: res.path } as Partial<CampaignSetupNodeData>);
      }
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Directory picker failed');
    } finally {
      setBusy(null);
    }
  }, [d.datasetPath, id, updateNodeData]);

  const scanDataset = useCallback(async () => {
    if (!d.datasetPath?.trim()) {
      setLocalError('Set a dataset root before scanning.');
      return;
    }

    setBusy('scan');
    setLocalError(null);
    try {
      const res = await api.scanCollaborationDataset(d.datasetPath);
      updateNodeData(id, {
        totalPatients: res.total_patients,
        patients: res.patients,
      } as Partial<CampaignSetupNodeData>);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Dataset scan failed');
    } finally {
      setBusy(null);
    }
  }, [d.datasetPath, id, updateNodeData]);

  const patientCount = d.campaign?.totalPatients ?? d.totalPatients ?? 0;
  const assigned = d.campaign?.progress.assignedPatients ?? 0;
  const completed = d.campaign?.progress.completed ?? 0;

  return (
    <BaseNode
      nodeId={id}
      nodeType="campaignSetup"
      title="Campaign Setup"
      icon="CS"
      color="var(--accent-blue)"
      status={d.status}
      error={d.error}
      hasInput={false}
      hasOutput={true}
      info={SETUP_INFO}
    >
      <label style={labelStyle}>Campaign Name</label>
      <input
        value={d.campaignName || ''}
        onChange={(event) => updateField('campaignName', event.target.value)}
        placeholder="e.g. Brain MRI Expert Review"
        style={inputStyle}
      />

      <label style={{ ...labelStyle, marginTop: 8 }}>Dataset Root</label>
      <div style={{ display: 'flex', gap: 5 }}>
        <input
          value={d.datasetPath || ''}
          onChange={(event) => updateField('datasetPath', event.target.value)}
          placeholder="C:/data/campaign_dataset"
          style={{ ...inputStyle, flex: 1, fontFamily: 'monospace' }}
        />
        <button type="button" onClick={pickDirectory} style={buttonStyle(busy === 'pick')} disabled={busy !== null}>
          Dir
        </button>
      </div>

      <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
        <button type="button" onClick={scanDataset} style={buttonStyle(busy === 'scan')} disabled={busy !== null}>
          Scan
        </button>
        <span style={{ color: 'var(--text-muted)', fontSize: 10, alignSelf: 'center' }}>
          Valid modalities: flair, t1, t1c, t2
        </span>
      </div>

      {patientCount > 0 ? (
        <div style={statStyle}>
          <div style={{ color: 'var(--accent-blue)', fontSize: 11, fontWeight: 800 }}>
            {patientCount} patients
          </div>
          <div style={{ color: 'var(--accent-green)', fontSize: 11, fontWeight: 800 }}>
            {assigned} assigned
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            {completed} completed
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            {d.campaign?.progress.reviewed ?? 0} reviewed
          </div>
        </div>
      ) : (
        <NodeHint>
          Scan the dataset first, then run the workflow to create or reuse the campaign.
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

export default memo(CampaignSetupNode);
