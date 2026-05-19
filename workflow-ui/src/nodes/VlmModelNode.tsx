/**
 * VLM model nodes
 * ===============
 * Shared React Flow node surface for MedGemma, SmolVLM, and Med-R1.
 *
 * These nodes do not embed prompt text. They fetch the same modality-specific
 * prompt presets used by the Gradio app through the FastAPI VLM router.
 */

import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { VlmModelId, VlmModality, VlmNodeData } from '../types/nodes';
import * as api from '../api/client';

const MODEL_META: Record<VlmModelId, {
  nodeType: string;
  title: string;
  icon: string;
  description: string;
  tips: string[];
}> = {
  medgemma: {
    nodeType: 'medgemmaNode',
    title: 'MedGemma',
    icon: 'MG',
    description: 'Runs MedGemma-4B with medical prompt presets from the Gradio app.',
    tips: [
      'Best for radiology-style descriptions and medical terminology.',
      'Use Label Suggester downstream when you want concise annotation labels.',
    ],
  },
  smolvlm: {
    nodeType: 'smolvlmNode',
    title: 'SmolVLM',
    icon: 'SV',
    description: 'Runs the lightweight SmolVLM model for quick slice captioning.',
    tips: [
      'Useful as a fast first-pass description node.',
      'For clinical wording and labels, compare with MedGemma or Med-R1.',
    ],
  },
  'med-r1': {
    nodeType: 'medR1Node',
    title: 'Med-R1',
    icon: 'R1',
    description: 'Runs Med-R1 for reasoning-oriented medical image review.',
    tips: [
      'Reasoning mode asks the model for a visible thinking/final-answer structure.',
      'Best suited to focused MRI review and region-level explanation.',
    ],
  },
};

const FALLBACK_PROMPTS = [
  { key: 'describe_slice', title: 'Describe Slice' },
  { key: 'identify_anomalies', title: 'Identify Anomalies' },
  { key: 'pathology_detection', title: 'Pathology Detection' },
  { key: 'structured_radiology_review', title: 'Structured Radiology Review' },
  { key: 'annotation_label_candidates', title: 'Annotation Label Candidates' },
];

const MODALITIES: VlmModality[] = ['MRI', 'CT', 'MG'];

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

const rowStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 86px',
  gap: 6,
  marginTop: 8,
};

const resultStyle: React.CSSProperties = {
  marginTop: 9,
  padding: '8px 9px',
  borderRadius: 6,
  border: '1px solid rgba(150, 115, 255, 0.28)',
  background: 'rgba(150, 115, 255, 0.08)',
};

const resultTextStyle: React.CSSProperties = {
  marginTop: 6,
  maxHeight: 130,
  overflowY: 'auto',
  whiteSpace: 'pre-wrap',
  fontSize: 10,
  lineHeight: 1.45,
  color: 'var(--text-secondary)',
};

const badgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '2px 7px',
  borderRadius: 4,
  border: '1px solid rgba(150, 115, 255, 0.38)',
  color: 'var(--accent-purple)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
};

function truncate(value: string, maxLength: number) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

function getPromptTitle(data: VlmNodeData) {
  const prompt = (data.availablePrompts || []).find((item) => item.key === data.promptKey);
  if (prompt) return prompt.title;
  const fallback = FALLBACK_PROMPTS.find((item) => item.key === data.promptKey);
  return fallback?.title || data.promptKey.replace(/_/g, ' ');
}

function VlmModelNode({ id, data, model }: NodeProps & { model: VlmModelId }) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as VlmNodeData;
  const meta = MODEL_META[model];
  const [promptError, setPromptError] = useState<string | null>(null);

  useEffect(() => {
    api.getVlmPrompts(d.modality || 'MRI')
      .then((res) => {
        setPromptError(null);
        updateNodeData(id, {
          availablePrompts: res.prompts,
        } as Partial<VlmNodeData>);
      })
      .catch((error) => {
        setPromptError(error instanceof Error ? error.message : 'Failed to load VLM prompts');
      });
  }, [d.modality, id, updateNodeData]);

  const info: NodeInfo = useMemo(() => ({
    description: meta.description,
    inputs: [
      'Loaded session from Data Loader or Format Converter',
      'Annotated session from Interactive Annotator when overlay-aware analysis is useful',
      'Optional voice prompt from Voice Input, which takes priority over the prompt preset',
    ],
    outputs: [
      'VLM analysis text with model, prompt, modality, and slice context',
    ],
    tips: meta.tips,
  }), [meta]);

  const availablePrompts = d.availablePrompts || [];
  const promptOptions = availablePrompts.length > 0
    ? availablePrompts.map((prompt) => ({ key: prompt.key, title: prompt.title }))
    : FALLBACK_PROMPTS;

  const handleSelect = useCallback(
    (key: keyof VlmNodeData) => (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateNodeData(id, { [key]: event.target.value } as Partial<VlmNodeData>);
    },
    [id, updateNodeData],
  );

  const handleNumber = useCallback(
    (key: keyof VlmNodeData) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const parsed = Number(event.target.value);
      updateNodeData(id, {
        [key]: Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
      } as Partial<VlmNodeData>);
    },
    [id, updateNodeData],
  );

  const handleText = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      updateNodeData(id, { customPrompt: event.target.value } as Partial<VlmNodeData>);
    },
    [id, updateNodeData],
  );

  const handleToggle = useCallback(
    (key: keyof VlmNodeData) => (event: React.ChangeEvent<HTMLInputElement>) => {
      updateNodeData(id, { [key]: event.target.checked } as Partial<VlmNodeData>);
    },
    [id, updateNodeData],
  );

  const result = d.vlmResult;
  const hasSession = Boolean(d.sessionId);
  const voicePrompt = d.voicePrompt;
  const selectedPromptTitle = voicePrompt?.text
    ? 'Voice Prompt'
    : d.customPrompt?.trim()
      ? 'Custom Prompt'
      : getPromptTitle(d);

  return (
    <BaseNode
      nodeId={id}
      nodeType={meta.nodeType}
      title={meta.title}
      icon={meta.icon}
      color="var(--accent-purple)"
      status={d.status}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={info}
    >
      <div style={{ fontSize: 11, color: hasSession ? 'var(--accent-green)' : 'var(--text-muted)' }}>
        {hasSession ? `Slice ${d.sliceIndex || 0} ready` : 'Awaiting Data Loader or Annotator'}
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
          Voice prompt: {truncate(voicePrompt.text, 110)}
        </div>
      ) : null}

      <div style={rowStyle}>
        <div>
          <label style={labelStyle}>Prompt</label>
          <select value={d.promptKey || 'describe_slice'} onChange={handleSelect('promptKey')} style={selectStyle}>
            {promptOptions.map((prompt) => (
              <option key={prompt.key} value={prompt.key}>{prompt.title}</option>
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

      <div style={rowStyle}>
        <div>
          <label style={labelStyle}>View</label>
          <select value={d.view || 'axial'} onChange={handleSelect('view')} style={selectStyle}>
            <option value="axial">Axial</option>
            <option value="coronal">Coronal</option>
            <option value="sagittal">Sagittal</option>
          </select>
        </div>
        <div>
          <label style={labelStyle}>Slice</label>
          <input
            type="number"
            min={0}
            value={Number(d.sliceIndex ?? 0)}
            onChange={handleNumber('sliceIndex')}
            style={inputStyle}
          />
        </div>
      </div>

      <details style={{ marginTop: 8 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 700 }}>
          Advanced prompt
        </summary>
        <textarea
          value={d.customPrompt || ''}
          onChange={handleText}
          placeholder="Optional custom prompt. Leave empty to use the selected preset."
          rows={4}
          style={{ ...inputStyle, resize: 'vertical', marginTop: 6, lineHeight: 1.35 }}
        />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
          <label style={{ ...labelStyle, display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={Boolean(d.useOverlay)}
              onChange={handleToggle('useOverlay')}
            />
            Use overlays
          </label>
          <label style={{ ...labelStyle, display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={Boolean(d.includeReasoning)}
              onChange={handleToggle('includeReasoning')}
              disabled={model !== 'med-r1'}
            />
            Reasoning
          </label>
        </div>
      </details>

      {promptError ? (
        <NodeHint>{promptError}. Fallback prompts are available.</NodeHint>
      ) : null}

      {result ? (
        <div style={resultStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
            <span style={badgeStyle}>{result.modelLabel}</span>
            <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
              {result.elapsedSeconds.toFixed(1)}s
            </span>
          </div>
          <div style={{ color: 'var(--text-primary)', fontSize: 11, fontWeight: 800, marginTop: 6 }}>
            {selectedPromptTitle}
          </div>
          <div style={resultTextStyle}>
            {truncate(result.text || 'No response text.', 900)}
          </div>
        </div>
      ) : (
        <NodeHint>
          Run this node to analyze the selected slice. Results can feed Label Suggester.
        </NodeHint>
      )}
    </BaseNode>
  );
}

function MedGemmaNode(props: NodeProps) {
  return <VlmModelNode {...props} model="medgemma" />;
}

function SmolVlmNode(props: NodeProps) {
  return <VlmModelNode {...props} model="smolvlm" />;
}

function MedR1Node(props: NodeProps) {
  return <VlmModelNode {...props} model="med-r1" />;
}

export const MedGemmaWorkflowNode = memo(MedGemmaNode);
export const SmolVlmWorkflowNode = memo(SmolVlmNode);
export const MedR1WorkflowNode = memo(MedR1Node);
