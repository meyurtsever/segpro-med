/**
 * LabelSuggesterNode
 * ==================
 * Produces concise annotation label candidates from a slice, annotated overlay,
 * or upstream VLM analysis.
 */

import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { InteractiveAnnotatorNodeData, LabelSuggestionDecision, LabelSuggesterNodeData, VlmModality, VlmModelId } from '../types/nodes';
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
    'Connect this directly to Interactive Annotator when label suggestions are the only VLM task.',
    'Place this after Interactive Annotator so the current slice drives suggestions without creating a graph cycle.',
  ],
};

const MODELS: Array<{ value: VlmModelId; label: string; tokens: number }> = [
  { value: 'medgemma', label: 'MedGemma', tokens: 256 },
  { value: 'medgemma-1.5', label: 'MedGemma 1.5', tokens: 256 },
  { value: 'medgemma-1.5-gguf', label: 'MedGemma 1.5 GGUF Q8', tokens: 512 },
  { value: 'smolvlm', label: 'SmolVLM', tokens: 128 },
  { value: 'med-r1', label: 'Med-R1', tokens: 384 },
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

const contextGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
  gap: 7,
  marginTop: 9,
};

const contextCardStyle: React.CSSProperties = {
  padding: '7px 8px',
  borderRadius: 5,
  border: '1px solid rgba(76, 175, 139, 0.24)',
  background: 'rgba(255, 255, 255, 0.035)',
};

const contextLabelStyle: React.CSSProperties = {
  display: 'block',
  color: 'var(--text-muted)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
  marginBottom: 3,
};

const contextValueStyle: React.CSSProperties = {
  color: 'var(--text-primary)',
  fontSize: 12,
  fontWeight: 800,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const chipWrapStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  marginTop: 9,
};

const chipStyle: React.CSSProperties = {
  padding: '5px 8px',
  borderRadius: 4,
  border: '1px solid rgba(76, 175, 139, 0.34)',
  background: 'rgba(76, 175, 139, 0.1)',
  color: 'var(--accent-green)',
  fontSize: 11,
  fontWeight: 800,
  lineHeight: 1.2,
};

const outputPanelStyle: React.CSSProperties = {
  width: '100%',
  marginTop: 10,
  padding: '11px 12px',
  borderRadius: 6,
  border: '1px solid rgba(76, 175, 139, 0.3)',
  background: 'rgba(76, 175, 139, 0.08)',
  textAlign: 'left',
  boxSizing: 'border-box',
};

const runButtonStyle: React.CSSProperties = {
  padding: '5px 11px',
  borderRadius: 5,
  border: '1px solid rgba(76, 175, 139, 0.42)',
  background: 'rgba(76, 175, 139, 0.12)',
  color: 'var(--accent-green)',
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
  width: 'min(760px, 92vw)',
  maxHeight: '84vh',
  borderRadius: 10,
  border: '1px solid rgba(76, 175, 139, 0.45)',
  background: 'var(--bg-secondary)',
  boxShadow: '0 24px 60px rgba(0, 0, 0, 0.38)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
};

function LabelSuggesterNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const storeNodes = useWorkflowStore((s) => s.nodes);
  const storeEdges = useWorkflowStore((s) => s.edges);
  const d = data as unknown as LabelSuggesterNodeData;
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [runHover, setRunHover] = useState(false);
  const [reviewLabel, setReviewLabel] = useState<string | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewMessage, setReviewMessage] = useState<string | null>(null);
  const [hoveredLabel, setHoveredLabel] = useState<string | null>(null);

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
      const value = event.target.value;
      const modelDefaults = key === 'model'
        ? MODELS.find((item) => item.value === value)
        : undefined;
      updateNodeData(id, {
        [key]: value,
        ...(modelDefaults ? { maxTokens: modelDefaults.tokens } : {}),
      } as Partial<LabelSuggesterNodeData>);
    },
    [id, updateNodeData],
  );

  const handleNumber = useCallback(
    (key: keyof LabelSuggesterNodeData) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const parsed = Number(event.target.value);
      updateNodeData(id, {
        [key]: Number.isFinite(parsed) ? Math.max(1, parsed) : 1,
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
  const voicePrompt = d.voicePrompt;
  const sliceIndex = Number(d.sliceIndex ?? 0);
  const view = d.view || 'axial';
  const sliceKey = `${view}:${sliceIndex}`;
  const suggestionContextMatches = !d.labelSuggestionContext ||
    (d.labelSuggestionContext.sliceIndex === sliceIndex && d.labelSuggestionContext.view === view);
  const suggestions = useMemo(
    () => suggestionContextMatches ? d.labelSuggestions || [] : [],
    [d.labelSuggestions, suggestionContextMatches],
  );
  const scaleFontStyle = {
    '--vlm-scale-font': 'clamp(13px, 1.55cqw, 17px)',
    '--vlm-small-font': 'clamp(11px, 1.2cqw, 14px)',
  } as React.CSSProperties;

  const upstreamHasSource = useMemo(() => {
    const upstreamIds = storeEdges.filter((e) => e.target === id).map((e) => e.source);
    return upstreamIds.some((upId) => {
      const upData = storeNodes.find((n) => n.id === upId)?.data as Record<string, unknown> | undefined;
      return Boolean(upData?.sessionId || upData?.vlmResult);
    });
  }, [id, storeNodes, storeEdges]);
  const hasSource = Boolean(d.sessionId || d.vlmResult) || upstreamHasSource;
  const isGenerating = d.status === 'running';
  const displayStatus = isGenerating
    ? 'waiting'
    : hasSource && d.status === 'waiting'
    ? 'idle'
    : (!hasSource && d.status === 'idle') ? 'waiting' : d.status;

  const upstreamAnnotator = useMemo(() => {
    const annotatorEdge = storeEdges.find((edge) => edge.target === id &&
      storeNodes.find((node) => node.id === edge.source)?.type === 'interactiveAnnotator');
    return annotatorEdge
      ? storeNodes.find((node) => node.id === annotatorEdge.source)
      : undefined;
  }, [id, storeEdges, storeNodes]);

  const handleLabelDecision = useCallback(
    async (decision: 'accepted' | 'rejected') => {
      if (!reviewLabel) return;
      const label = reviewLabel.trim();
      const studyPath = typeof d.sourcePath === 'string' ? d.sourcePath.trim() : '';
      if (!studyPath) {
        setReviewMessage('Load path unavailable. Run Data Loader before saving labels.');
        return;
      }

      setReviewBusy(true);
      setReviewMessage(null);

      try {
        const res = await api.reviewVlmLabelSuggestion({
          user_id: typeof d.userId === 'string' && d.userId.trim() ? d.userId.trim() : 'workflow_user',
          study_path: studyPath,
          slice_idx: sliceIndex,
          view_type: view,
          label,
          action: decision,
          suggested_labels: suggestions,
          source: 'workflow',
        });

        const nextSuggestions = suggestions.filter((candidate) =>
          candidate.trim().toLowerCase() !== label.toLowerCase(),
        );
        const nextSuggestionsBySlice = {
          ...(d.labelSuggestionsBySlice || {}),
          [sliceKey]: nextSuggestions,
        };
        const priorDecisions = (d.labelSuggestionDecisions || []) as LabelSuggestionDecision[];
        const nextDecision: LabelSuggestionDecision = {
          label,
          decision,
          sliceIndex,
          view,
          decidedAt: new Date().toISOString(),
          source: 'vlm',
        };
        const nextDecisions = [
          ...priorDecisions.filter((item) =>
            !(item.sliceIndex === sliceIndex && item.view === view && item.label.trim().toLowerCase() === label.toLowerCase()),
          ),
          nextDecision,
        ];

        updateNodeData(id, {
          labelSuggestions: nextSuggestions,
          labelSuggestionsBySlice: nextSuggestionsBySlice,
          labelSuggestionDecisions: nextDecisions,
          savedLabels: res.saved_labels,
        } as Partial<LabelSuggesterNodeData>);

        if (upstreamAnnotator) {
          const annotatorData = upstreamAnnotator.data as InteractiveAnnotatorNodeData;
          const annotatorDecisions = (annotatorData.labelSuggestionDecisions || []) as LabelSuggestionDecision[];
          const mergedAnnotatorDecisions = [
            ...annotatorDecisions.filter((item) =>
              !(item.sliceIndex === sliceIndex && item.view === view && item.label.trim().toLowerCase() === label.toLowerCase()),
            ),
            nextDecision,
          ];
          updateNodeData(upstreamAnnotator.id, {
            labelSuggestions: nextSuggestions,
            labelSuggestionResult: d.labelSuggestionResult,
            labelSuggestionContext: d.labelSuggestionContext || { sliceIndex, view },
            labelSuggestionDecisions: mergedAnnotatorDecisions,
            savedLabels: res.saved_labels,
          } as Partial<InteractiveAnnotatorNodeData>);
        }

        setReviewLabel(null);
        setReviewMessage(res.message);
      } catch (error) {
        setReviewMessage(error instanceof Error ? error.message : 'Could not save label decision');
      } finally {
        setReviewBusy(false);
      }
    },
    [
      d.labelSuggestionDecisions,
      d.labelSuggestionContext,
      d.labelSuggestionResult,
      d.labelSuggestionsBySlice,
      d.sourcePath,
      d.userId,
      id,
      reviewLabel,
      sliceIndex,
      sliceKey,
      suggestions,
      updateNodeData,
      upstreamAnnotator,
      view,
    ],
  );

  return (
    <BaseNode
      nodeId={id}
      nodeType="labelSuggester"
      title="Label Suggester"
      icon="LS"
      color="var(--accent-green)"
      status={displayStatus}
      error={d.error}
      hasInput={true}
      hasOutput={true}
      info={LABEL_INFO}
      footerAction={hasSource ? (
        <button
          type="button"
          style={{
            ...runButtonStyle,
            background: runHover ? 'rgba(76, 175, 139, 0.2)' : runButtonStyle.background,
          }}
          onMouseEnter={() => setRunHover(true)}
          onMouseLeave={() => setRunHover(false)}
          onClick={() => {
            window.dispatchEvent(new CustomEvent('segpro:run-node', { detail: { nodeId: id } }));
          }}
        >
          Suggest Labels
        </button>
      ) : null}
      resizable={true}
      minWidth={430}
      minHeight={470}
    >
      {(!hasSource || isGenerating) && (
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
              ? 'The label model is reviewing the active slice. Suggestions will appear here when complete.'
              : 'Run the Data Loader and inspect the slice in Interactive Annotator before asking for label suggestions.'}
          </div>
        </div>
      )}

      <div style={{ ...scaleFontStyle, opacity: hasSource && !isGenerating ? 1 : 0.22 }}>
        <div style={{ fontSize: 'var(--vlm-scale-font)', color: hasSource ? 'var(--accent-green)' : 'var(--text-muted)', fontWeight: 800 }}>
          {hasSource ? 'Active annotator context' : 'No image context yet'}
        </div>

        <div style={contextGridStyle}>
          <div style={contextCardStyle}>
            <span style={contextLabelStyle}>View</span>
            <span style={contextValueStyle}>{d.view || 'axial'}</span>
          </div>
          <div style={contextCardStyle}>
            <span style={contextLabelStyle}>Slice</span>
            <span style={contextValueStyle}>{sliceIndex + 1}</span>
          </div>
          <div style={contextCardStyle}>
            <span style={contextLabelStyle}>Modality</span>
            <span style={contextValueStyle}>{d.modality || 'MRI'}</span>
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
            Voice prompt connected
          </div>
        ) : null}

        <div style={{ marginTop: 11 }}>
          <label style={{ ...labelStyle, fontSize: 'var(--vlm-scale-font)', color: 'var(--text-primary)', fontWeight: 800 }}>Prompt</label>
          <select
            value={d.promptKey || 'suggest_labels'}
            onChange={handleSelect('promptKey')}
            style={{ ...selectStyle, padding: '10px 11px', fontSize: 'var(--vlm-scale-font)' }}
          >
            {safePromptOptions.map((prompt) => (
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
          <div style={twoColStyle}>
            <div>
              <label style={labelStyle}>Model</label>
              <select value={d.model || 'medgemma-1.5-gguf'} onChange={handleSelect('model')} style={selectStyle}>
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
          <div style={twoColStyle}>
            <div>
              <label style={labelStyle}>Tokens</label>
              <input
                type="number"
                min={32}
                max={1024}
                value={Number(d.maxTokens ?? 512)}
                onChange={handleNumber('maxTokens')}
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Max labels</label>
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
        </details>

        {suggestions.length > 0 ? (
          <div style={outputPanelStyle}>
            <div style={{
              marginBottom: 9,
              padding: '8px 10px',
              borderRadius: 6,
              border: '1px solid rgba(76, 175, 139, 0.34)',
              background: 'rgba(76, 175, 139, 0.1)',
              color: 'var(--accent-green)',
              fontSize: 'var(--vlm-small-font)',
              fontWeight: 800,
              lineHeight: 1.4,
            }}>
              Labels are ready for this {view} slice {sliceIndex + 1}. Select a label to approve or refuse it.
            </div>
            <div style={{
              color: 'var(--text-primary)',
              fontSize: 'var(--vlm-scale-font)',
              fontWeight: 900,
            }}>
              {suggestions.length} suggestion{suggestions.length === 1 ? '' : 's'}
            </div>
            <div style={chipWrapStyle}>
              {suggestions.map((label) => (
                <button
                  key={label}
                  type="button"
                  style={{
                    ...chipStyle,
                    borderColor: hoveredLabel === label ? 'rgba(76, 175, 139, 0.72)' : 'rgba(76, 175, 139, 0.34)',
                    background: hoveredLabel === label ? 'rgba(76, 175, 139, 0.2)' : 'rgba(76, 175, 139, 0.1)',
                    cursor: 'pointer',
                  }}
                  onMouseEnter={() => setHoveredLabel(label)}
                  onMouseLeave={() => setHoveredLabel(null)}
                  onClick={() => {
                    setReviewLabel(label);
                    setReviewMessage(null);
                  }}
                  title="Review suggested label"
                >
                  {label}
                </button>
              ))}
            </div>
            {reviewMessage ? (
              <div style={{
                marginTop: 8,
                color: reviewMessage.toLowerCase().includes('could') || reviewMessage.toLowerCase().includes('unavailable')
                  ? 'var(--accent-red)'
                  : 'var(--accent-green)',
                fontSize: 11,
                lineHeight: 1.4,
              }}>
                {reviewMessage}
              </div>
            ) : null}
          </div>
        ) : (
          <NodeHint>
            Run this node after inspecting the selected slice. Suggestions are bound to the current slice.
          </NodeHint>
        )}
      </div>

      {reviewLabel ? createPortal(
        <div style={modalBackdropStyle} onClick={() => !reviewBusy && setReviewLabel(null)}>
          <div style={modalPanelStyle} onClick={(event) => event.stopPropagation()}>
            <div style={{
              padding: '14px 16px',
              borderBottom: '1px solid rgba(76, 175, 139, 0.28)',
              display: 'flex',
              justifyContent: 'space-between',
              gap: 12,
              alignItems: 'center',
            }}>
              <div>
                <div style={{ color: 'var(--accent-green)', fontSize: 13, fontWeight: 900 }}>
                  Review Label
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginTop: 3 }}>
                  {view} slice {sliceIndex + 1}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setReviewLabel(null)}
                disabled={reviewBusy}
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: 6,
                  border: '1px solid var(--border-color)',
                  background: 'transparent',
                  color: 'var(--text-secondary)',
                  cursor: reviewBusy ? 'not-allowed' : 'pointer',
                  fontWeight: 900,
                }}
                title="Close"
              >
                x
              </button>
            </div>
            <div style={{ padding: 16 }}>
              <div style={{
                padding: '12px 14px',
                borderRadius: 6,
                border: '1px solid rgba(76, 175, 139, 0.32)',
                background: 'rgba(76, 175, 139, 0.08)',
                color: 'var(--accent-green)',
                fontSize: 16,
                fontWeight: 900,
              }}>
                {reviewLabel}
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.5, marginTop: 10 }}>
                Accepting seals this label to the current {view} slice {sliceIndex + 1}.
              </div>
              {reviewMessage ? (
                <div style={{ marginTop: 10, color: 'var(--accent-red)', fontSize: 12 }}>
                  {reviewMessage}
                </div>
              ) : null}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
                <button
                  type="button"
                  disabled={reviewBusy}
                  onClick={() => handleLabelDecision('rejected')}
                  style={{
                    padding: '7px 12px',
                    borderRadius: 6,
                    border: '1px solid var(--border-color)',
                    background: 'transparent',
                    color: 'var(--text-secondary)',
                    cursor: reviewBusy ? 'wait' : 'pointer',
                    fontWeight: 800,
                  }}
                >
                  Refuse
                </button>
                <button
                  type="button"
                  disabled={reviewBusy}
                  onClick={() => handleLabelDecision('accepted')}
                  style={{
                    padding: '7px 12px',
                    borderRadius: 6,
                    border: '1px solid rgba(76, 175, 139, 0.42)',
                    background: 'rgba(76, 175, 139, 0.16)',
                    color: 'var(--accent-green)',
                    cursor: reviewBusy ? 'wait' : 'pointer',
                    fontWeight: 900,
                  }}
                >
                  {reviewBusy ? 'Saving...' : 'Accept'}
                </button>
              </div>
            </div>
          </div>
        </div>,
        document.body,
      ) : null}
    </BaseNode>
  );
}

export default memo(LabelSuggesterNode);
