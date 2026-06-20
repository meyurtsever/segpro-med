/**
 * VLM model nodes
 * ===============
 * Shared React Flow node surface for medical report generation.
 *
 * These nodes do not embed prompt text. They fetch the same modality-specific
 * prompt presets used by the Gradio app through the FastAPI VLM router.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { VlmModelId, VlmModality, VlmNodeData } from '../types/nodes';
import * as api from '../api/client';
import {
  getSpeechRecognitionConstructor,
  type SpeechRecognitionLike,
} from '../utils/browserDictation';

const MODEL_META: Record<VlmModelId, {
  nodeType: string;
  title: string;
  icon: string;
  description: string;
  tips: string[];
}> = {
  medgemma: {
    nodeType: 'medgemmaNode',
    title: 'Medical Report Generation',
    icon: 'MR',
    description: 'Runs MedGemma-4B with medical prompt presets from the Gradio app.',
    tips: [
      'Best for radiology-style descriptions and medical terminology.',
      'Use Label Suggester downstream when you want concise annotation labels.',
    ],
  },
  'medgemma-1.5': {
    nodeType: 'medgemmaNode',
    title: 'Medical Report Generation',
    icon: 'M1',
    description: 'Runs the locally installed MedGemma 1.5 4B model through Transformers.',
    tips: [
      'Best default when you want the newer MedGemma checkpoint on the RTX 5080.',
      'Uses deterministic decoding for repeatable reports and better latency.',
    ],
  },
  'medgemma-1.5-gguf': {
    nodeType: 'medgemmaNode',
    title: 'Medical Report Generation',
    icon: 'GQ',
    description: 'Runs the local MedGemma 1.5 Q8 GGUF model through llama.cpp when a GPU backend is installed.',
    tips: [
      'Requires llama-cpp-python with GPU support; CPU-only llama.cpp is not recommended.',
      'Q8 preserves quality better than smaller quantizations while reducing model size.',
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
const MODELS: Array<{ value: VlmModelId; label: string; tokens: number; reasoning: boolean }> = [
  { value: 'medgemma', label: 'MedGemma', tokens: 256, reasoning: false },
  { value: 'medgemma-1.5', label: 'MedGemma 1.5', tokens: 256, reasoning: false },
  { value: 'medgemma-1.5-gguf', label: 'MedGemma 1.5 GGUF Q8', tokens: 512, reasoning: false },
  { value: 'smolvlm', label: 'SmolVLM', tokens: 128, reasoning: false },
  { value: 'med-r1', label: 'Med-R1', tokens: 384, reasoning: true },
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
  fontSize: 13,
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

const resultButtonStyle: React.CSSProperties = {
  width: '100%',
  marginTop: 10,
  padding: '11px 12px',
  borderRadius: 6,
  border: '1px solid rgba(150, 115, 255, 0.28)',
  background: 'rgba(150, 115, 255, 0.08)',
  textAlign: 'left',
  cursor: 'pointer',
  boxSizing: 'border-box',
  transition: 'border-color 120ms ease, background 120ms ease',
};

const resultTextStyle: React.CSSProperties = {
  marginTop: 8,
  minHeight: 145,
  maxHeight: 210,
  overflowY: 'auto',
  whiteSpace: 'pre-wrap',
  fontSize: 13,
  lineHeight: 1.5,
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

const contextGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: 8,
  marginTop: 10,
};

const contextCardStyle = (color: string): React.CSSProperties => ({
  minHeight: 62,
  padding: '9px 10px',
  borderRadius: 7,
  border: `1px solid color-mix(in srgb, ${color} 38%, var(--border-color))`,
  background: `color-mix(in srgb, ${color} 12%, var(--bg-secondary))`,
  boxSizing: 'border-box',
});

const contextLabelStyle: React.CSSProperties = {
  display: 'block',
  color: 'var(--text-muted)',
  fontSize: 9,
  fontWeight: 900,
  textTransform: 'uppercase',
  marginBottom: 5,
};

const contextValueStyle = (color: string, fontSize = 20): React.CSSProperties => ({
  color,
  fontSize,
  fontWeight: 950,
  lineHeight: 1.08,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
});

const runButtonStyle: React.CSSProperties = {
  padding: '5px 11px',
  borderRadius: 5,
  border: '1px solid rgba(150, 115, 255, 0.42)',
  background: 'rgba(150, 115, 255, 0.12)',
  color: 'var(--accent-purple)',
  fontSize: 12,
  fontWeight: 900,
  cursor: 'pointer',
};

const modalBackdropStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 10000,
  background: 'rgba(8, 10, 24, 0.72)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 24,
};

const modalPanelStyle: React.CSSProperties = {
  width: 'min(860px, 92vw)',
  maxHeight: '84vh',
  borderRadius: 10,
  border: '1px solid rgba(150, 115, 255, 0.45)',
  background: 'var(--bg-secondary)',
  boxShadow: '0 24px 60px rgba(0, 0, 0, 0.38)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
};

const modalHeaderButtonStyle = (active = false, disabled = false): React.CSSProperties => ({
  height: 30,
  borderRadius: 6,
  border: `1px solid ${active ? 'rgba(76, 175, 139, 0.58)' : 'rgba(76, 175, 139, 0.34)'}`,
  background: active ? 'rgba(76, 175, 139, 0.18)' : 'rgba(76, 175, 139, 0.08)',
  color: active ? 'var(--accent-green)' : 'var(--accent-green)',
  cursor: disabled ? 'not-allowed' : active ? 'progress' : 'pointer',
  fontSize: 11,
  fontWeight: 900,
  padding: '0 10px',
  opacity: disabled ? 0.45 : 1,
  transition: 'background 0.15s ease, border-color 0.15s ease',
});

const modalHeaderInfoStyle: React.CSSProperties = {
  color: 'var(--accent-green)',
  fontSize: 11,
  lineHeight: 1.35,
  maxWidth: 280,
  padding: '6px 9px',
  borderRadius: 6,
  border: '1px solid rgba(76, 175, 139, 0.32)',
  background: 'rgba(76, 175, 139, 0.12)',
  fontWeight: 800,
};

const reportEditorStyle: React.CSSProperties = {
  width: '100%',
  minHeight: 380,
  resize: 'vertical',
  border: '1px solid rgba(150, 115, 255, 0.24)',
  borderRadius: 8,
  background: 'rgba(255, 255, 255, 0.035)',
  color: 'var(--text-primary)',
  fontSize: 14,
  lineHeight: 1.55,
  padding: 12,
  boxSizing: 'border-box',
  outline: 'none',
};

const markdownContainerStyle: React.CSSProperties = {
  whiteSpace: 'normal',
  color: 'inherit',
};

const markdownParagraphStyle: React.CSSProperties = {
  margin: '0 0 8px',
};

const markdownListStyle: React.CSSProperties = {
  margin: '6px 0 10px',
  paddingLeft: 20,
};

const markdownListItemStyle: React.CSSProperties = {
  marginBottom: 5,
};

const boxedAnswerStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  minHeight: 22,
  padding: '1px 8px',
  margin: '0 2px',
  border: '1px solid rgba(88, 166, 255, 0.72)',
  borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)',
  color: '#8fbcff',
  fontWeight: 900,
};

function truncate(value: string, maxLength: number) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(\$\\boxed\{[^}]+\}\$|\\boxed\{[^}]+\}|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    const boxedMatch = /^\$?\\boxed\{([^}]+)\}\$?$/.exec(token);
    if (boxedMatch) {
      parts.push(
        <span key={`${match.index}-box`} style={boxedAnswerStyle}>
          {boxedMatch[1]}
        </span>,
      );
    } else if (token.startsWith('**')) {
      parts.push(
        <strong key={`${match.index}-b`}>
          {token.slice(2, -2)}
        </strong>,
      );
    } else {
      parts.push(
        <em key={`${match.index}-i`}>
          {token.slice(1, -1)}
        </em>,
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

function renderReportMarkdown(text: string) {
  const lines = text.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let listItems: ReactNode[] = [];

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`list-${blocks.length}`} style={markdownListStyle}>
        {listItems}
      </ul>,
    );
    listItems = [];
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      return;
    }

    const bulletMatch = /^[-*]\s+(.+)$/.exec(line);
    if (bulletMatch) {
      listItems.push(
        <li key={`li-${index}`} style={markdownListItemStyle}>
          {renderInlineMarkdown(bulletMatch[1])}
        </li>,
      );
      return;
    }

    flushList();
    blocks.push(
      <p key={`p-${index}`} style={markdownParagraphStyle}>
        {renderInlineMarkdown(line)}
      </p>,
    );
  });

  flushList();
  return <div style={markdownContainerStyle}>{blocks}</div>;
}

function getPromptTitle(data: VlmNodeData) {
  const prompt = (data.availablePrompts || []).find((item) => item.key === data.promptKey);
  if (prompt) return prompt.title;
  const fallback = FALLBACK_PROMPTS.find((item) => item.key === data.promptKey);
  return fallback?.title || data.promptKey.replace(/_/g, ' ');
}

function VlmModelNode({ id, data, model }: NodeProps & { model: VlmModelId }) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const storeNodes = useWorkflowStore((s) => s.nodes);
  const storeEdges = useWorkflowStore((s) => s.edges);
  const d = data as unknown as VlmNodeData;
  const selectedModel = d.model || 'medgemma-1.5-gguf';
  const meta = model === 'medgemma' ? MODEL_META.medgemma : MODEL_META[selectedModel];
  const selectedModelDefaults = MODELS.find((item) => item.value === selectedModel);
  const [promptError, setPromptError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [resultOpen, setResultOpen] = useState(false);
  const [resultHover, setResultHover] = useState(false);
  const [runHover, setRunHover] = useState(false);
  const [editedReportText, setEditedReportText] = useState('');
  const [editingReport, setEditingReport] = useState(false);
  const [dictating, setDictating] = useState(false);
  const [dictationError, setDictationError] = useState<string | null>(null);
  const [editButtonHover, setEditButtonHover] = useState(false);
  const [dictateButtonHover, setDictateButtonHover] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const lastDictationAppendRef = useRef('');

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
      if (key === 'model') {
        const nextModel = event.target.value as VlmModelId;
        const modelDefaults = MODELS.find((item) => item.value === nextModel);
        updateNodeData(id, {
          model: nextModel,
          maxTokens: modelDefaults?.tokens,
          includeReasoning: modelDefaults?.reasoning,
        } as Partial<VlmNodeData>);
        return;
      }
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

  const sliceIndex = Number(d.sliceIndex ?? 0);
  const view = d.view || 'axial';
  const result = d.vlmResult &&
    d.vlmResult.sliceIndex === sliceIndex &&
    d.vlmResult.view === view
    ? d.vlmResult
    : undefined;
  const voicePrompt = d.voicePrompt;
  const isGenerating = d.status === 'running';

  useEffect(() => () => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
  }, []);

  const appendReportText = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || lastDictationAppendRef.current === trimmed) return;
    lastDictationAppendRef.current = trimmed;
    setEditedReportText((current) => {
      const separator = current.trim() ? '\n\n' : '';
      return `${current}${separator}${trimmed}`;
    });
  }, []);

  const toggleModalDictation = useCallback(() => {
    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setDictationError('Browser dictation is not available in this browser.');
      return;
    }

    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
      setDictating(false);
      return;
    }

    const recognition = new Recognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      const chunks: string[] = [];
      for (let index = 0; index < event.results.length; index += 1) {
        if (event.results[index].isFinal) {
          chunks.push(event.results[index][0].transcript);
        }
      }
      appendReportText(chunks.join(' '));
    };
    recognition.onerror = (event) => {
      setDictationError(event.error ? `Dictation failed: ${event.error}` : 'Dictation failed');
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setDictating(false);
    };

    setDictationError(null);
    lastDictationAppendRef.current = '';
    recognitionRef.current = recognition;
    setDictating(true);
    try {
      recognition.start();
    } catch (error) {
      recognitionRef.current = null;
      setDictating(false);
      setDictationError(error instanceof Error ? error.message : 'Dictation could not start');
    }
  }, [appendReportText]);

  const upstreamHasSession = useMemo(() => {
    const upstreamIds = storeEdges.filter((e) => e.target === id).map((e) => e.source);
    return upstreamIds.some((upId) => {
      const upData = storeNodes.find((n) => n.id === upId)?.data as Record<string, unknown> | undefined;
      return Boolean(upData?.sessionId);
    });
  }, [id, storeNodes, storeEdges]);
  const hasSession = Boolean(d.sessionId) || upstreamHasSession;

  const selectedPromptTitle = voicePrompt?.text
    ? 'Voice Prompt'
    : d.customPrompt?.trim()
      ? 'Custom Prompt'
      : getPromptTitle(d);
  const selectedModelLabel = MODELS.find((item) => item.value === selectedModel)?.label ||
    selectedModel;

  const displayStatus = isGenerating
    ? 'waiting'
    : hasSession && d.status === 'waiting'
    ? 'idle'
    : (!hasSession && d.status === 'idle') ? 'waiting' : d.status;
  const scaleFontStyle = {
    '--vlm-scale-font': 'clamp(13px, 1.9cqh, 18px)',
    '--vlm-small-font': 'clamp(11px, 1.45cqh, 14px)',
  } as React.CSSProperties;

  return (
    <BaseNode
      nodeId={id}
      nodeType={meta.nodeType}
      title={meta.title}
      icon={meta.icon}
      color="var(--accent-purple)"
      status={displayStatus}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={info}
      footerAction={hasSession ? (
        <button
          type="button"
          style={{
            ...runButtonStyle,
            background: runHover ? 'rgba(150, 115, 255, 0.2)' : runButtonStyle.background,
          }}
          onMouseEnter={() => setRunHover(true)}
          onMouseLeave={() => setRunHover(false)}
          onClick={() => {
            window.dispatchEvent(new CustomEvent('segpro:run-node', { detail: { nodeId: id } }));
          }}
        >
          Create Report
        </button>
      ) : null}
      resizable={true}
      minWidth={460}
      maxWidth={460}
      minHeight={470}
    >
      {(!hasSession || isGenerating) && (
        <div
          style={{
            minHeight: 250,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            padding: '18px 20px',
            borderRadius: 8,
            border: '1px solid rgba(245, 181, 79, 0.38)',
            background: 'rgba(245, 181, 79, 0.08)',
            marginBottom: 10,
          }}
        >
          <div style={{ color: 'var(--accent-orange)', fontSize: 28, fontWeight: 900, marginBottom: 8 }}>
            ◌
          </div>
          <div style={{ color: 'var(--accent-orange)', fontSize: 16, fontWeight: 900, marginBottom: 6 }}>
            {isGenerating ? 'Generating a response...' : 'Load image first'}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.5 }}>
            {isGenerating
              ? 'The selected medical VLM is analyzing the active slice. The report will appear here when complete.'
              : 'Run the Data Loader and inspect the slice in Interactive Annotator. This node will wait until that context exists.'}
          </div>
        </div>
      )}

      <div style={{ ...scaleFontStyle, opacity: hasSession && !isGenerating ? 1 : 0.22 }}>
        <div style={{ fontSize: 'var(--vlm-scale-font)', color: hasSession ? 'var(--accent-green)' : 'var(--text-muted)', fontWeight: 800 }}>
          {hasSession ? 'Active annotator context' : 'No image context yet'}
        </div>

        <div style={contextGridStyle}>
          <div style={contextCardStyle('var(--accent-purple)')}>
            <span style={contextLabelStyle}>View</span>
            <span style={contextValueStyle('var(--accent-purple)')}>{view}</span>
          </div>
          <div style={contextCardStyle('var(--accent-blue)')}>
            <span style={contextLabelStyle}>Slice</span>
            <span style={contextValueStyle('var(--accent-blue)')}>{sliceIndex + 1}</span>
          </div>
          <div style={contextCardStyle('var(--accent-green)')}>
            <span style={contextLabelStyle}>Model</span>
            <span style={contextValueStyle('var(--accent-green)', 14)} title={selectedModelLabel}>
              {selectedModelLabel}
            </span>
          </div>
          <div style={contextCardStyle('var(--accent-orange)')}>
            <span style={contextLabelStyle}>Prompt</span>
            <span style={contextValueStyle('var(--accent-orange)', 14)} title={selectedPromptTitle}>
              {selectedPromptTitle}
            </span>
          </div>
        </div>

        {voicePrompt?.text ? (
          <div
            style={{
              marginTop: 8,
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

        <div style={{ marginTop: 11 }}>
          <label style={{ ...labelStyle, fontSize: 'var(--vlm-scale-font)', color: 'var(--text-primary)', fontWeight: 800 }}>Prompt</label>
          <select
            value={d.promptKey || 'describe_slice'}
            onChange={handleSelect('promptKey')}
            style={{ ...selectStyle, padding: '10px 11px', fontSize: 'var(--vlm-scale-font)' }}
          >
            {promptOptions.map((prompt) => (
              <option key={prompt.key} value={prompt.key}>{prompt.title}</option>
            ))}
          </select>
        </div>

        <details
          open={settingsOpen}
          onToggle={(event) => setSettingsOpen(event.currentTarget.open)}
          style={{ marginTop: 9 }}
        >
          <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', fontSize: 11, fontWeight: 800 }}>
            Settings
          </summary>
          <div style={rowStyle}>
            <div>
              <label style={labelStyle}>Model</label>
              <select value={selectedModel} onChange={handleSelect('model')} style={selectStyle}>
                {MODELS.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
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
              <label style={labelStyle}>Tokens</label>
              <input
                type="number"
                min={32}
                max={2048}
                value={Number(d.maxTokens ?? selectedModelDefaults?.tokens ?? 512)}
                onChange={handleNumber('maxTokens')}
                style={inputStyle}
              />
            </div>
            <div />
          </div>
          <textarea
            value={d.customPrompt || ''}
            onChange={handleText}
            placeholder="Optional custom prompt. Leave empty to use the selected preset."
            rows={4}
            style={{ ...inputStyle, resize: 'vertical', marginTop: 8, lineHeight: 1.35 }}
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
                disabled={selectedModel !== 'med-r1'}
              />
              Reasoning
            </label>
          </div>
        </details>

        {promptError ? (
          <NodeHint>{promptError}. Fallback prompts are available.</NodeHint>
        ) : null}

        {result ? (
          <button
            type="button"
            style={{
              ...resultButtonStyle,
              borderColor: resultHover ? 'rgba(150, 115, 255, 0.58)' : 'rgba(150, 115, 255, 0.28)',
              background: resultHover ? 'rgba(150, 115, 255, 0.14)' : 'rgba(150, 115, 255, 0.08)',
            }}
            onClick={() => {
              setEditedReportText(result.text || '');
              setDictationError(null);
              setEditingReport(false);
              lastDictationAppendRef.current = '';
              setResultOpen(true);
            }}
            onMouseEnter={() => setResultHover(true)}
            onMouseLeave={() => setResultHover(false)}
            title="Open full VLM output"
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
              <span style={badgeStyle}>{result.modelLabel}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                {result.elapsedSeconds.toFixed(1)}s
              </span>
            </div>
            <div style={{ color: 'var(--text-primary)', fontSize: 'var(--vlm-scale-font)', fontWeight: 800, marginTop: 7 }}>
              {selectedPromptTitle}
            </div>
            <div style={resultTextStyle}>
              {renderReportMarkdown(truncate(result.text || 'No response text.', 900))}
            </div>
          </button>
        ) : (
          <NodeHint>
            Run this node after inspecting the selected slice. The output opens in a larger modal.
          </NodeHint>
        )}
      </div>

      {resultOpen && result ? createPortal(
        <div
          style={modalBackdropStyle}
          onClick={() => {
            recognitionRef.current?.stop();
            recognitionRef.current = null;
            setDictating(false);
            setResultOpen(false);
          }}
        >
          <div style={modalPanelStyle} onClick={(event) => event.stopPropagation()}>
            <div style={{
              padding: '14px 16px',
              borderBottom: '1px solid rgba(150, 115, 255, 0.28)',
              display: 'flex',
              justifyContent: 'space-between',
              gap: 12,
              alignItems: 'center',
            }}>
              <div>
                <div style={{ color: 'var(--accent-purple)', fontSize: 13, fontWeight: 900 }}>
                  {meta.title} Output
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginTop: 3 }}>
                  {selectedPromptTitle} - {result.view} slice {Number(result.sliceIndex ?? 0) + 1}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={modalHeaderInfoStyle}>
                  Add notes by dictating or switch Edit on to revise the report.
                </div>
                <button
                  type="button"
                  onClick={() => setEditingReport((value) => !value)}
                  style={{
                    ...modalHeaderButtonStyle(editingReport),
                    background: editButtonHover
                      ? 'rgba(76, 175, 139, 0.22)'
                      : modalHeaderButtonStyle(editingReport).background,
                    border: editButtonHover
                      ? '1px solid rgba(76, 175, 139, 0.72)'
                      : modalHeaderButtonStyle(editingReport).border,
                  }}
                  onMouseEnter={() => setEditButtonHover(true)}
                  onMouseLeave={() => setEditButtonHover(false)}
                  title={editingReport ? 'Show rendered report' : 'Edit report text'}
                >
                  {editingReport ? 'Save changes' : 'Edit'}
                </button>
                <button
                  type="button"
                  onClick={toggleModalDictation}
                  style={{
                    ...modalHeaderButtonStyle(dictating),
                    background: dictateButtonHover
                      ? 'rgba(76, 175, 139, 0.22)'
                      : modalHeaderButtonStyle(dictating).background,
                    border: dictateButtonHover
                      ? '1px solid rgba(76, 175, 139, 0.72)'
                      : modalHeaderButtonStyle(dictating).border,
                  }}
                  onMouseEnter={() => setDictateButtonHover(true)}
                  onMouseLeave={() => setDictateButtonHover(false)}
                  title="Append dictated text to the report"
                >
                  {dictating ? 'Listening...' : 'Dictate'}
                </button>
              <button
                type="button"
                onClick={() => {
                  recognitionRef.current?.stop();
                  recognitionRef.current = null;
                  setDictating(false);
                  setResultOpen(false);
                }}
                style={modalHeaderButtonStyle()}
                title="Close"
              >
                x
              </button>
              </div>
            </div>
            <div style={{
              padding: 16,
              overflowY: 'auto',
              color: 'var(--text-primary)',
              fontSize: 14,
              lineHeight: 1.55,
            }}>
              {editingReport ? (
                <textarea
                  value={editedReportText}
                  onChange={(event) => setEditedReportText(event.target.value)}
                  style={reportEditorStyle}
                  aria-label="Editable medical report"
                />
              ) : (
                <div style={{
                  ...reportEditorStyle,
                  overflowY: 'auto',
                  resize: 'none',
                }}>
                  {renderReportMarkdown(editedReportText || 'No response text.')}
                </div>
              )}
              <div style={{
                marginTop: 8,
                color: dictationError ? 'var(--accent-red)' : 'var(--text-muted)',
                fontSize: 11,
                lineHeight: 1.4,
              }}>
                {dictationError ||
                  (dictating
                    ? 'Listening. Dictated text will be appended to the end of the report.'
                    : editingReport
                      ? 'Editing is on. Changes are kept inside this modal for review.'
                      : 'Rendered preview is shown. Use Edit to revise text, or Dictate to append notes.')}
              </div>
            </div>
          </div>
        </div>,
        document.body,
      ) : null}
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
