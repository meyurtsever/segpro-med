/**
 * VoiceInputNode
 * ==============
 * Captures a spoken or typed instruction and emits a voicePrompt contract for
 * MedGemma, SmolVLM, Med-R1, and Label Suggester nodes.
 */

import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { type NodeProps } from '@xyflow/react';

import BaseNode from './BaseNode';
import NodeHint from '../components/NodeHint';
import type { NodeInfo } from '../components/InfoModal';
import useWorkflowStore from '../store/workflowStore';
import type { VoiceInputNodeData, VoicePromptIntent } from '../types/nodes';
import * as api from '../api/client';

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  0: { transcript: string };
};

type SpeechRecognitionEventLike = {
  results: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

const VOICE_INFO: NodeInfo = {
  description:
    'Turns spoken instructions into a VLM prompt. It follows the Gradio voice behavior: anomaly words route to anomaly prompts, description words route to descriptive prompts, and unclear instructions default to both.',
  inputs: [],
  outputs: ['Voice prompt text and inferred intent for medical VLM analysis nodes'],
  tips: [
    'Use this upstream of MedGemma, SmolVLM, Med-R1, or Label Suggester.',
    'The transcript remains editable so the clinician can correct speech recognition before running the VLM.',
    'Uploaded audio is transcribed through the backend Whisper helper when local dependencies are available.',
  ],
};

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

const buttonStyle = (active = false): React.CSSProperties => ({
  padding: '6px 8px',
  borderRadius: 5,
  border: `1px solid ${active ? 'var(--accent-orange)' : 'var(--border-color)'}`,
  background: active ? 'rgba(245, 181, 79, 0.14)' : 'var(--bg-tertiary)',
  color: active ? 'var(--accent-orange)' : 'var(--text-secondary)',
  fontSize: 10,
  fontWeight: 800,
  cursor: active ? 'progress' : 'pointer',
});

const pillStyle = (intent: VoicePromptIntent): React.CSSProperties => {
  const color = intent === 'anomaly'
    ? 'var(--accent-red)'
    : intent === 'describe'
      ? 'var(--accent-blue)'
      : intent === 'custom'
        ? 'var(--text-secondary)'
        : 'var(--accent-orange)';

  return {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '3px 7px',
    borderRadius: 4,
    border: `1px solid color-mix(in srgb, ${color} 44%, var(--border-color))`,
    background: `color-mix(in srgb, ${color} 12%, var(--bg-tertiary))`,
    color,
    fontSize: 10,
    fontWeight: 800,
    textTransform: 'uppercase',
  };
};

function getSpeechRecognitionConstructor(): SpeechRecognitionConstructor | undefined {
  const scopedWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return scopedWindow.SpeechRecognition || scopedWindow.webkitSpeechRecognition;
}

function mapVoiceResponse(res: api.VoicePromptResponse) {
  return {
    transcript: res.transcript,
    intent: res.intent as VoicePromptIntent,
    promptKey: res.prompt_key,
    voicePrompt: {
      text: res.transcript,
      intent: res.intent as VoicePromptIntent,
      promptKey: res.prompt_key,
      identifyAnomalies: res.identify_anomalies,
      describeSlice: res.describe_slice,
      source: res.source === 'audio_path' || res.source === 'upload' || res.source === 'dictation' || res.source === 'typed'
        ? res.source
        : 'manual',
    },
  } as Partial<VoiceInputNodeData>;
}

function VoiceInputNode({ id, data }: NodeProps) {
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData);
  const d = data as unknown as VoiceInputNodeData;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const [busy, setBusy] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const speechSupported = useMemo(
    () => typeof window !== 'undefined' && Boolean(getSpeechRecognitionConstructor()),
    [],
  );

  const updateTranscript = useCallback(
    (value: string) => {
      updateNodeData(id, {
        transcript: value,
        voicePrompt: value.trim()
          ? {
            text: value.trim(),
            intent: d.intent || 'both',
            promptKey: d.promptKey || 'structured_radiology_review',
            identifyAnomalies: d.intent === 'anomaly' || d.intent === 'both',
            describeSlice: d.intent === 'describe' || d.intent === 'both',
            source: 'typed',
          }
          : undefined,
      } as Partial<VoiceInputNodeData>);
    },
    [d.intent, d.promptKey, id, updateNodeData],
  );

  const detectIntent = useCallback(async () => {
    if (!d.transcript?.trim()) {
      setLocalError('Add a transcript before detecting intent.');
      return;
    }

    setBusy(true);
    setLocalError(null);
    try {
      const res = await api.inferVoiceIntent(d.transcript);
      updateNodeData(id, mapVoiceResponse(res));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Intent detection failed');
    } finally {
      setBusy(false);
    }
  }, [d.transcript, id, updateNodeData]);

  const startDictation = useCallback(() => {
    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setLocalError('Browser dictation is not available in this browser.');
      return;
    }

    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
      setRecognizing(false);
      return;
    }

    const recognition = new Recognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      const chunks: string[] = [];
      for (let index = 0; index < event.results.length; index += 1) {
        chunks.push(event.results[index][0].transcript);
      }
      const transcript = chunks.join(' ').trim();
      if (transcript) {
        updateNodeData(id, {
          transcript,
          voicePrompt: {
            text: transcript,
            intent: d.intent || 'both',
            promptKey: d.promptKey || 'structured_radiology_review',
            identifyAnomalies: d.intent === 'anomaly' || d.intent === 'both',
            describeSlice: d.intent === 'describe' || d.intent === 'both',
            source: 'dictation',
          },
        } as Partial<VoiceInputNodeData>);
      }
    };
    recognition.onerror = (event) => {
      setLocalError(event.error ? `Dictation failed: ${event.error}` : 'Dictation failed');
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setRecognizing(false);
    };

    setLocalError(null);
    recognitionRef.current = recognition;
    setRecognizing(true);
    recognition.start();
  }, [d.intent, d.promptKey, id, updateNodeData]);

  const transcribeFile = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setBusy(true);
      setLocalError(null);
      updateNodeData(id, { audioFileName: file.name } as Partial<VoiceInputNodeData>);

      try {
        const res = await api.transcribeVoiceFile(file);
        updateNodeData(id, {
          ...mapVoiceResponse(res),
          audioFileName: file.name,
        });
      } catch (error) {
        setLocalError(error instanceof Error ? error.message : 'Voice transcription failed');
      } finally {
        setBusy(false);
        event.target.value = '';
      }
    },
    [id, updateNodeData],
  );

  const transcriptPreview = d.transcript?.trim();

  return (
    <BaseNode
      nodeId={id}
      nodeType="voiceInput"
      title="Voice Input"
      icon="VO"
      color="var(--accent-orange)"
      status={d.status}
      error={d.error}
      hasInput={false}
      hasOutput={true}
      info={VOICE_INFO}
    >
      <label style={labelStyle}>Transcript / Prompt</label>
      <textarea
        value={d.transcript || ''}
        onChange={(event) => updateTranscript(event.target.value)}
        placeholder="Example: describe this slice and point out possible lesions"
        rows={4}
        style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.35 }}
      />

      <div style={{ display: 'flex', gap: 6, marginTop: 7, flexWrap: 'wrap' }}>
        <button
          type="button"
          onClick={startDictation}
          style={buttonStyle(recognizing)}
          disabled={!speechSupported}
          title={speechSupported ? 'Use browser speech recognition' : 'Browser dictation is unavailable'}
        >
          {recognizing ? 'Stop' : 'Dictate'}
        </button>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          style={buttonStyle(busy)}
          disabled={busy}
          title="Upload an audio file and transcribe it through the backend"
        >
          Audio
        </button>
        <button
          type="button"
          onClick={detectIntent}
          style={buttonStyle(busy)}
          disabled={busy || !transcriptPreview}
          title="Infer prompt intent using the Gradio voice rules"
        >
          Intent
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={transcribeFile}
        style={{ display: 'none' }}
      />

      <div style={{ display: 'flex', gap: 5, alignItems: 'center', marginTop: 8, flexWrap: 'wrap' }}>
        <span style={pillStyle(d.intent || 'both')}>{d.intent || 'both'}</span>
        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
          {d.promptKey || 'structured_radiology_review'}
        </span>
      </div>

      {d.audioFileName ? (
        <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 10, wordBreak: 'break-word' }}>
          Audio: {d.audioFileName}
        </div>
      ) : null}

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
      ) : !transcriptPreview ? (
        <NodeHint>
          Add a transcript, dictate, or upload audio. The connected VLM node will use this prompt first.
        </NodeHint>
      ) : null}
    </BaseNode>
  );
}

export default memo(VoiceInputNode);
