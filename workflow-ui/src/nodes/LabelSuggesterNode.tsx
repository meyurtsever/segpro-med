/**
 * LabelSuggesterNode
 * ==================
 * Produces concise annotation label candidates from a slice, annotated overlay,
 * or upstream VLM analysis.
 */

import { memo, useCallback, useEffect } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { LabelSuggesterNodeData, VlmModality, VlmModelId } from '../types/nodes';
import * as api from '../api/client';

const LABEL_INFO: NodeInfo = {
  description:
    'Generates short label candidates for annotation work. It can run a VLM on the current slice or parse an upstream VLM analysis.',
  inputs: [
    'Loaded session from Data Loader or Format Converter',
    'Annotated session from Interactive Annotator for overlay-aware suggestions',
    'Optional VLM analysis from MedGemma, SmolVLM, or Med-R1',
    'Optional voice prompt from Voice Input for focused label requests',
  ],
  outputs: [
    'Label suggestions that can be reviewed in the node or connected to Interactive Annotator',
  ],
  tips: [
    'Use overlay mode when the annotator has drawn regions that need semantic labels.',
    'MedGemma is the default because its Gradio label prompts are the most specialized.',
    'Accepted label persistence belongs to the later Label Management pass.',
  ],
};

const MODELS: Array<{ value: VlmModelId; label: string }> = [
  { value: 'medgemma', label: 'MedGemma' },
  { value: 'smolvlm', label: 'SmolVLM' },
  { value: 'med-r1', label: 'Med-R1' },
];

const MODALITIES: VlmModality[] = ['MRI', 'CT', 'MG'];

const FALLBACK_PROMPTS = [
  { key: 'suggest_labels', title: 'Suggest Labels' },
  { key: 'annotation_label_candidates', title: 'Annotation Label Candidates' },
  { key: 'mri_label_suggestions', title: 'MRI Label Suggestions' },
  { key: 'ct_label_suggestions', title: 'CT Label Suggestions' },
  { key: 'mg_label_suggestions', title: 'Mammography Label Suggestions' },
];

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
  display: 'block',
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

const twoColStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 84px',
  gap: 6,
  marginTop: 8,
};

const chipWrapStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 5,
  marginTop: 8,
};

const chipStyle: React.CSSProperties = {
  padding: '4px 7px',
  borderRadius: 4,
  border: '1px solid rgba(76, 175, 139, 0.34)',
  background: 'rgba(76, 175, 139, 0.1)',
  color: 'var(--accent-green)',
  fontSize: 10,
  fontWeight: 800,
  lineHeight: 1.2,
};

function LabelSuggesterNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as LabelSuggesterNodeData;

  useEffect(() => {
    api.getVlmPrompts(d.modality || 'MRI')
      .then((res) => {
        updateNodeData(id, {
          availablePrompts: res.prompts,
        } as Partial<LabelSuggesterNodeData>);
      })
      .catch(() => {
        // Fallback prompts are enough for the node to remain configurable.
      });
  }, [d.modality, id, updateNodeData]);

  const handleSelect = useCallback(
    (key: keyof LabelSuggesterNodeData) => (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { [key]: event.target.value } as Partial<LabelSuggesterNodeData>);
    },
    [id, updateNodeData],
  );

  const handleNumber = useCallback(
    (key: keyof LabelSuggesterNodeData) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const parsed = Number(event.target.value);
      updateNodeData(id, {
        [key]: Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
      } as Partial<LabelSuggesterNodeData>);
    },
    [id, updateNodeData],
  );

  const handleToggle = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { useOverlay: event.target.checked } as Partial<LabelSuggesterNodeData>);
    },
    [id, updateNodeData],
  );

  const availablePrompts = d.availablePrompts || [];
  const promptOptions = availablePrompts.length > 0
    ? availablePrompts
        .filter((prompt) =>
          prompt.key.includes('label') ||
          prompt.key === 'suggest_labels' ||
          prompt.key.includes('organ'),
        )
        .map((prompt) => ({ key: prompt.key, title: prompt.title }))
    : FALLBACK_PROMPTS;
  const safePromptOptions = promptOptions.length > 0 ? promptOptions : FALLBACK_PROMPTS;
  const suggestions = d.labelSuggestions || [];
  const hasSource = Boolean(d.sessionId || d.vlmResult);
  const voicePrompt = d.voicePrompt;

  return (
    <BaseNode
      nodeId={id}
      nodeType="labelSuggester"
      title="Label Suggester"
      icon="LS"
      color="var(--accent-green)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={LABEL_INFO}
    >
      <div style={{ fontSize: 11, color: hasSource ? 'var(--accent-green)' : 'var(--text-muted)' }}>
        {hasSource ? 'Ready to suggest labels' : 'Awaiting image or VLM analysis'}
      </div>

      {voicePrompt?.text ? (
        <div
          style={{
            marginTop: 7,
            padding: '6px 8px',
            borderRadius: 5,
            border: '1px solid rgba(245, 181, 79, 0.3)',
            background: 'rgba(245, 181, 79, 0.08)',
            color: 'var(--accent-orange)',
            fontSize: 10,
            lineHeight: 1.35,
          }}
        >
          Voice prompt connected
        </div>
      ) : null}

      <div style={twoColStyle}>
        <div>
          <label style={labelStyle}>Model</label>
          <select value={d.model || 'medgemma'} onChange={handleSelect('model')} style={selectStyle}>
            {MODELS.map((model) => (
              <option key={model.value} value={model.value}>{model.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={labelStyle}>Modality</label>
          <select value={d.modality || 'MRI'} onChange={handleSelect('modality')} style={selectStyle}>
            {MODALITIES.map((modality) => (
              <option key={modality} value={modality}>{modality}</option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <label style={labelStyle}>Prompt</label>
        <select value={d.promptKey || 'suggest_labels'} onChange={handleSelect('promptKey')} style={selectStyle}>
          {safePromptOptions.map((prompt) => (
            <option key={prompt.key} value={prompt.key}>{prompt.title}</option>
          ))}
        </select>
      </div>

      <div style={twoColStyle}>
        <div>
          <label style={labelStyle}>View</label>
          <select value={d.view || 'axial'} onChange={handleSelect('view')} style={selectStyle}>
            <option value="axial">Axial</option>
            <option value="coronal">Coronal</option>
            <option value="sagittal">Sagittal</option>
          </select>
        </div>
        <div>
          <label style={labelStyle}>Max</label>
          <input
            type="number"
            min={1}
            max={32}
            value={Number(d.maxLabels ?? 12)}
            onChange={handleNumber('maxLabels')}
            style={inputStyle}
          />
        </div>
      </div>

      <label style={{ ...labelStyle, display: 'flex', gap: 6, alignItems: 'center', marginTop: 8 }}>
        <input type="checkbox" checked={d.useOverlay !== false} onChange={handleToggle} />
        Use annotation overlays when available
      </label>

      {suggestions.length > 0 ? (
        <>
          <div style={{
            marginTop: 8,
            color: 'var(--text-primary)',
            fontSize: 11,
            fontWeight: 800,
          }}>
            {suggestions.length} suggestion{suggestions.length === 1 ? '' : 's'}
          </div>
          <div style={chipWrapStyle}>
            {suggestions.map((label) => (
              <span key={label} style={chipStyle}>{label}</span>
            ))}
          </div>
        </>
      ) : (
        <NodeHint>
          Run this node to create concise label chips. Connect it to Interactive Annotator for review.
        </NodeHint>
      )}
    </BaseNode>
  );
}

export default memo(LabelSuggesterNode);
