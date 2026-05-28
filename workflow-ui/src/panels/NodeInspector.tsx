/**
 * NodeInspector
 * =============
 * Right-side configuration and guidance panel for selected workflow nodes.
 */

import { useCallback, useMemo, useState, type CSSProperties } from 'react';
import type { Node } from '@xyflow/react';

import useWorkflowStore from '../store/workflowStore';
import { nodePaletteItems } from '../nodes';
import { getAllowedSources, getAllowedTargets } from '../engine/compatibility';
import { getPortDefinition, type WorkflowPortKind } from '../engine/nodeContracts';
import type { BaseNodeData } from '../types/nodes';
import * as api from '../api/client';

const PANEL_WIDTH = 320;
const COLLAPSED_WIDTH = 44;

const panelStyle: CSSProperties = {
  width: PANEL_WIDTH,
  flexShrink: 0,
  height: '100%',
  background: 'var(--bg-secondary)',
  borderLeft: '1px solid var(--border-color)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  transition: 'width 0.18s ease',
};

const headerStyle: CSSProperties = {
  padding: '16px 16px 12px',
  borderBottom: '1px solid var(--border-color)',
};

const bodyStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: 12,
};

const collapsedBodyStyle: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '8px 0',
};

const sectionStyle: CSSProperties = {
  marginBottom: 14,
  paddingBottom: 14,
  borderBottom: '1px solid var(--border-color)',
};

const sectionTitleStyle: CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  textTransform: 'uppercase',
  letterSpacing: 0.8,
  color: 'var(--text-muted)',
  marginBottom: 8,
};

const labelStyle: CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: 'var(--text-secondary)',
  marginBottom: 4,
};

const inputStyle: CSSProperties = {
  width: '100%',
  padding: '7px 8px',
  background: 'var(--bg-tertiary)',
  border: '1px solid var(--border-color)',
  borderRadius: 5,
  color: 'var(--text-primary)',
  fontSize: 12,
  outline: 'none',
  boxSizing: 'border-box',
};

const monoInputStyle: CSSProperties = {
  ...inputStyle,
  fontFamily: 'monospace',
};

const rowStyle: CSSProperties = {
  marginBottom: 10,
};

const inputRowStyle: CSSProperties = {
  display: 'flex',
  gap: 4,
  alignItems: 'stretch',
};

const miniButtonStyle = (active = false): CSSProperties => ({
  minWidth: 38,
  padding: '7px 7px',
  borderRadius: 5,
  border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border-color)'}`,
  background: active ? 'rgba(79, 141, 245, 0.16)' : 'var(--bg-tertiary)',
  color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
  fontSize: 10,
  fontWeight: 800,
  cursor: active ? 'progress' : 'pointer',
  flexShrink: 0,
});

const pillStyle = (color: string): CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'center',
  padding: '5px 8px',
  borderRadius: 4,
  border: `1px solid color-mix(in srgb, ${color} 55%, var(--border-color))`,
  background: `color-mix(in srgb, ${color} 16%, var(--bg-tertiary))`,
  color,
  fontSize: 10,
  fontWeight: 700,
  lineHeight: 1.15,
  whiteSpace: 'nowrap',
});

const helpTextStyle: CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 11,
  lineHeight: 1.5,
};

const connectionRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
  padding: '6px 0',
  fontSize: 11,
  color: 'var(--text-secondary)',
};

const collapseButtonStyle: CSSProperties = {
  width: 28,
  height: 28,
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  fontSize: 13,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

const infoBoxStyle = (color: string): CSSProperties => ({
  padding: '8px 9px',
  borderRadius: 7,
  border: `1px solid color-mix(in srgb, ${color} 45%, var(--border-color))`,
  background: `color-mix(in srgb, ${color} 10%, var(--bg-secondary))`,
  marginTop: 7,
});

const infoBoxTitleStyle = (color: string): CSSProperties => ({
  fontSize: 10,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0.7,
  color,
  marginBottom: 5,
});

const chipWrapStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 5,
};

const connectionSectionStyle: CSSProperties = {
  marginBottom: 14,
  padding: 12,
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--accent-blue) 28%, var(--border-color))',
  borderLeft: '3px solid var(--accent-blue)',
  background: 'rgba(79, 141, 245, 0.06)',
};

const dividerStyle: CSSProperties = {
  height: 1,
  background: 'var(--border-color)',
  opacity: 0.85,
  margin: '10px 0 0',
};

const statusColor: Record<string, string> = {
  idle: 'var(--text-muted)',
  running: 'var(--accent-orange)',
  success: 'var(--accent-green)',
  error: 'var(--accent-red)',
};

const CONVERSION_TYPES = [
  'DICOM to NIFTI',
  'NIFTI to MAT',
  'DICOM to MAT',
  'NIFTI to PNG',
];

const PNG_AXIS_OPTIONS = [
  { value: 2, label: 'Axial (axis 2)' },
  { value: 1, label: 'Coronal (axis 1)' },
  { value: 0, label: 'Sagittal (axis 0)' },
];

const FALLBACK_SEGMENTATION_CONFIGS = [
  'fast',
  'balanced',
  'high_detail',
  'small_structures',
  'tumor_detection',
  'skull_stripping',
  'mammography',
];

const SAM2_RUN_MODES = [
  'single',
  'range',
  'wholeVolume',
];

const SAM2_PROMPT_MODES = [
  'auto',
  'prompt',
];

const ANNOTATION_TOOLS = [
  'rect',
  'polygon',
  'circle',
  'freehand',
  'point',
  'pan',
  'eraser',
];

const VLM_MODALITIES = ['MRI', 'CT', 'MG'];

const VLM_MODELS = ['medgemma', 'medgemma-1.5', 'medgemma-1.5-gguf', 'smolvlm', 'med-r1'];

const FALLBACK_VLM_PROMPTS = [
  'describe_slice',
  'identify_anomalies',
  'pathology_detection',
  'structured_radiology_review',
  'annotation_label_candidates',
];

const FALLBACK_LABEL_PROMPTS = [
  'suggest_labels',
  'annotation_label_candidates',
  'mri_label_suggestions',
  'ct_label_suggestions',
  'mg_label_suggestions',
];

const VOICE_INTENTS = [
  'describe',
  'anomaly',
  'both',
  'custom',
];

const VOICE_PROMPT_KEYS = [
  'describe_slice',
  'identify_anomalies',
  'structured_radiology_review',
  'pathology_detection',
];

const ASSIGNMENT_MODES = [
  'allUnassigned',
  'selected',
];

function getNodeTitle(node: Node<BaseNodeData> | undefined): string {
  if (!node) return 'Unknown node';
  const paletteItem = nodePaletteItems.find((item) => item.type === node.type);
  return paletteItem?.label || node.data.label || node.type || node.id;
}

function formatNodeType(type: string | undefined): string {
  const paletteItem = nodePaletteItems.find((item) => item.type === type);
  return paletteItem?.label || type || 'Unknown';
}

export default function NodeInspector() {
  const [collapsed, setCollapsed] = useState(false);
  const [pickerBusy, setPickerBusy] = useState<string | null>(null);
  const {
    nodes,
    edges,
    selectedNodeId,
    selectedEdgeId,
    updateNodeData,
    removeNode,
    removeEdge,
    setSelectedNodeId,
    setSelectedEdgeId,
    setWorkflowNotice,
  } = useWorkflowStore();

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedNodeId),
    [nodes, selectedNodeId],
  );

  const selectedEdge = useMemo(
    () => edges.find((edge) => edge.id === selectedEdgeId),
    [edges, selectedEdgeId],
  );

  const selectedData = selectedNode?.data as Record<string, unknown> | undefined;

  const incoming = useMemo(
    () => edges.filter((edge) => edge.target === selectedNodeId),
    [edges, selectedNodeId],
  );

  const outgoing = useMemo(
    () => edges.filter((edge) => edge.source === selectedNodeId),
    [edges, selectedNodeId],
  );

  const updateField = (key: string, value: unknown) => {
    if (!selectedNode) return;
    updateNodeData(selectedNode.id, { [key]: value });
  };

  const openNativePathPicker = useCallback(
    async (key: string, mode: 'file' | 'directory') => {
      if (!selectedNode) return;

      const busyKey = `${key}:${mode}`;
      const currentValue = String(selectedData?.[key] ?? '');
      setPickerBusy(busyKey);

      try {
        const result = await api.openNativePathDialog(mode, currentValue);
        if (!result.cancelled && result.path) {
          updateNodeData(selectedNode.id, { [key]: result.path });
        }
      } catch (err) {
        setWorkflowNotice({
          type: 'error',
          message: err instanceof Error
            ? err.message
            : 'Native file selector could not be opened.',
        });
      } finally {
        setPickerBusy(null);
      }
    },
    [selectedData, selectedNode, setWorkflowNotice, updateNodeData],
  );

  const panelShellStyle: CSSProperties = {
    ...panelStyle,
    width: collapsed ? COLLAPSED_WIDTH : PANEL_WIDTH,
  };

  const renderCollapseButton = (title: string) => (
    <button
      onClick={() => setCollapsed((value) => !value)}
      style={collapseButtonStyle}
      title={title}
      aria-label={title}
    >
      {collapsed ? '<' : '>'}
    </button>
  );

  const renderCompatibilityBox = (
    title: string,
    values: string[],
    emptyLabel: string,
    color: string,
  ) => (
    <div style={infoBoxStyle(color)}>
      <div style={infoBoxTitleStyle(color)}>{title}</div>
      {values.length > 0 ? (
        <div style={chipWrapStyle}>
          {values.map((value) =>
            renderNodeChip(formatNodeType(value), color),
          )}
        </div>
      ) : (
        <div style={helpTextStyle}>{emptyLabel}</div>
      )}
    </div>
  );

  const renderNodeChip = (label: string, color: string) => (
    <span key={label} style={pillStyle(color)}>
      <span>{label}</span>
    </span>
  );

  const renderConnectionBox = (
    title: string,
    connectionNodes: Array<{ id: string; node: Node<BaseNodeData> | undefined }>,
    color: string,
  ) => {
    if (connectionNodes.length === 0) return null;

    return (
      <div style={infoBoxStyle(color)}>
        <div style={infoBoxTitleStyle(color)}>{title}</div>
        <div style={chipWrapStyle}>
          {connectionNodes.map(({ id, node }) =>
            <span key={node?.id || id}>
              {renderNodeChip(getNodeTitle(node), color)}
            </span>,
          )}
        </div>
      </div>
    );
  };

  const renderTextField = (
    label: string,
    key: string,
    placeholder = '',
    readOnly = false,
  ) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <input
          value={String(selectedData[key] ?? '')}
          onChange={(event) => updateField(key, event.target.value)}
          placeholder={placeholder}
          readOnly={readOnly}
          style={{
            ...(readOnly ? monoInputStyle : inputStyle),
            opacity: readOnly ? 0.75 : 1,
          }}
        />
      </div>
    );
  };

  const renderTextareaField = (
    label: string,
    key: string,
    placeholder = '',
  ) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <textarea
          value={String(selectedData[key] ?? '')}
          onChange={(event) => updateField(key, event.target.value)}
          placeholder={placeholder}
          rows={4}
          style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.4 }}
        />
      </div>
    );
  };

  const renderPathField = (
    label: string,
    key: string,
    placeholder = '',
    modes: Array<'file' | 'directory'> = ['directory'],
  ) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <div style={inputRowStyle}>
          <input
            value={String(selectedData[key] ?? '')}
            onChange={(event) => updateField(key, event.target.value)}
            placeholder={placeholder}
            style={{ ...inputStyle, flex: 1 }}
          />
          {modes.map((mode) => {
            const busyKey = `${key}:${mode}`;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => openNativePathPicker(key, mode)}
                disabled={pickerBusy !== null}
                style={miniButtonStyle(pickerBusy === busyKey)}
                title={mode === 'file' ? 'Select file' : 'Select directory'}
              >
                {mode === 'file' ? 'File' : 'Dir'}
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  const renderSelectField = (
    label: string,
    key: string,
    options: string[],
    placeholder = 'Select...',
  ) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <select
          value={String(selectedData[key] ?? '')}
          onChange={(event) => updateField(key, event.target.value)}
          style={inputStyle}
        >
          <option value="">{placeholder}</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>
    );
  };

  const renderNumberField = (label: string, key: string) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <input
          type="number"
          value={Number(selectedData[key] ?? 0)}
          onChange={(event) => updateField(key, Number(event.target.value))}
          style={inputStyle}
        />
      </div>
    );
  };

  const renderNumberSelectField = (
    label: string,
    key: string,
    options: Array<{ value: number; label: string }>,
  ) => {
    if (!selectedData) return null;
    return (
      <div style={rowStyle}>
        <label style={labelStyle}>{label}</label>
        <select
          value={Number(selectedData[key] ?? options[0]?.value ?? 0)}
          onChange={(event) => updateField(key, Number(event.target.value))}
          style={inputStyle}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    );
  };

  const renderCheckboxField = (label: string, key: string) => {
    if (!selectedData) return null;
    return (
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: 'var(--text-secondary)',
          fontSize: 12,
          marginBottom: 8,
        }}
      >
        <input
          type="checkbox"
          checked={Boolean(selectedData[key])}
          onChange={(event) => updateField(key, event.target.checked)}
        />
        {label}
      </label>
    );
  };

  const renderConfig = () => {
    if (!selectedNode || !selectedData) return null;

    if (selectedNode.type === 'dataLoader') {
      return (
        <>
          {renderPathField(
            'File or Directory Path',
            'path',
            'C:/data/patient01 or /data/brain.nii.gz',
            ['file', 'directory'],
          )}
          {renderPathField('Dataset Search Root', 'searchRoot', 'Optional root directory')}
        </>
      );
    }

    if (selectedNode.type === 'formatConverter') {
      return (
        <>
          {renderSelectField('Conversion Type', 'conversionType', CONVERSION_TYPES)}
          {renderTextField('Input Path', 'inputPath', 'Auto-filled from upstream when possible')}
          {renderTextField('Output Path', 'outputPath', 'Auto-generated if empty')}
          {selectedData.conversionType === 'DICOM to NIFTI'
            ? renderCheckboxField('Compress NIfTI output (.nii.gz)', 'compress')
            : null}
          {selectedData.conversionType === 'NIFTI to PNG'
            ? renderNumberSelectField('PNG Slice Axis', 'axis', PNG_AXIS_OPTIONS)
            : null}
        </>
      );
    }

    if (selectedNode.type === 'sliceViewer') {
      return (
        <>
          {renderSelectField('View Plane', 'view', ['axial', 'coronal', 'sagittal'])}
          {renderNumberField('Slice Index', 'sliceIndex')}
          {selectedData.segPath ? renderTextField('Segmentation Path', 'segPath', '', true) : null}
        </>
      );
    }

    if (selectedNode.type === 'metadataViewer') {
      return (
        <>
          {renderPathField(
            'Standalone Source Path',
            'sourcePath',
            'Optional DICOM directory, NIfTI, or MAT file',
            ['file', 'directory'],
          )}
          {renderTextField('Filter Text', 'filterText', 'Optional metadata filter')}
        </>
      );
    }

    if (selectedNode.type === 'deidentifyNode') {
      return (
        <>
          {renderPathField(
            'Source Path',
            'sourcePath',
            'Auto-filled from Data Loader when connected',
            ['file', 'directory'],
          )}
          {renderPathField(
            'Output Path',
            'outputPath',
            'Optional. Auto-generated if empty',
            ['file', 'directory'],
          )}
          {renderCheckboxField('Apply pydeface to 3D volume when supported', 'applyDeface')}
          {selectedData.auditPath ? renderTextField('Audit Path', 'auditPath', '', true) : null}
        </>
      );
    }

    if (selectedNode.type === 'interactiveAnnotator') {
      return (
        <>
          {renderSelectField('Active Tool', 'activeTool', ANNOTATION_TOOLS)}
          {renderCheckboxField('Show labels on canvas', 'showLabels')}
          {renderNumberField('Slice Index', 'sliceIndex')}
        </>
      );
    }

    if (selectedNode.type === 'autoSegmentation') {
      const apiConfigs = Array.isArray(selectedData.availableConfigs)
        ? (selectedData.availableConfigs as Array<{ name?: string }>)
            .map((config) => config.name)
            .filter((name): name is string => Boolean(name))
        : [];

      return (
        <>
          {renderSelectField(
            'Segmentation Profile',
            'configName',
            apiConfigs.length > 0 ? apiConfigs : FALLBACK_SEGMENTATION_CONFIGS,
          )}
        </>
      );
    }

    if (selectedNode.type === 'medsam2Segmenter') {
      const apiConfigs = Array.isArray(selectedData.availableConfigs)
        ? (selectedData.availableConfigs as Array<{ name?: string }>)
            .map((config) => config.name)
            .filter((name): name is string => Boolean(name))
        : [];

      return (
        <>
          {renderSelectField('Prompt Mode', 'promptMode', SAM2_PROMPT_MODES)}
          {renderSelectField('Run Mode', 'runMode', SAM2_RUN_MODES)}
          {renderSelectField(
            'Segmentation Profile',
            'configName',
            apiConfigs.length > 0 ? apiConfigs : FALLBACK_SEGMENTATION_CONFIGS,
          )}
          {renderSelectField('View Plane', 'view', ['axial', 'coronal', 'sagittal'])}
          {selectedData.runMode === 'range' ? (
            <>
              {renderNumberField('Start Slice', 'sliceStart')}
              {renderNumberField('End Slice', 'sliceEnd')}
              {renderNumberField('Slice Step', 'sliceStep')}
            </>
          ) : selectedData.runMode === 'wholeVolume' ? (
            <>
              {renderNumberField('Slice Step', 'sliceStep')}
            </>
          ) : (
            renderNumberField('Slice Index', 'sliceIndex')
          )}
        </>
      );
    }

    if (selectedNode.type === 'annotationStore') {
      return (
        <>
          {renderTextField('User ID', 'userId', 'workflow_user')}
          {renderPathField('Study Path', 'studyPath', 'Source study path', ['file', 'directory'])}
          {renderSelectField('Annotation Type', 'annotationType', [
            'manual',
            'guided_segmentation',
            'whole_area_segmentation',
          ])}
        </>
      );
    }

    if (selectedNode.type === 'annotationLoad') {
      return (
        <>
          {renderTextField('User ID', 'userId', 'workflow_user')}
          {renderPathField('Study Path', 'studyPath', 'Source study path', ['file', 'directory'])}
        </>
      );
    }

    if (selectedNode.type === 'exportNode') {
      return (
        <>
          {renderTextField('User ID', 'userId', 'workflow_user')}
          {renderPathField('Study Path', 'studyPath', 'Source study path', ['file', 'directory'])}
          {renderSelectField('Export Format', 'exportFormat', ['json'])}
          {selectedData.outputPath ? renderTextField('Output Path', 'outputPath', '', true) : null}
        </>
      );
    }

    if (selectedNode.type === 'medgemmaNode' ||
      selectedNode.type === 'smolvlmNode' ||
      selectedNode.type === 'medR1Node') {
      const apiPrompts = Array.isArray(selectedData.availablePrompts)
        ? (selectedData.availablePrompts as Array<{ key?: string }>)
            .map((prompt) => prompt.key)
            .filter((key): key is string => Boolean(key))
        : [];

      return (
        <>
          {renderSelectField(
            'Prompt Preset',
            'promptKey',
            apiPrompts.length > 0 ? apiPrompts : FALLBACK_VLM_PROMPTS,
          )}
          {renderSelectField('Model', 'model', ['medgemma', 'smolvlm', 'med-r1'])}
          {renderSelectField('Modality', 'modality', VLM_MODALITIES)}
          {renderSelectField('View Plane', 'view', ['axial', 'coronal', 'sagittal'])}
          {renderNumberField('Slice Index', 'sliceIndex')}
          {renderNumberField('Max Tokens', 'maxTokens')}
          {selectedNode.type === 'medR1Node' || selectedData.model === 'med-r1'
            ? renderCheckboxField('Ask Med-R1 for reasoning format', 'includeReasoning')
            : null}
          {renderCheckboxField('Use annotation overlays when available', 'useOverlay')}
          {renderTextareaField(
            'Custom Prompt',
            'customPrompt',
            'Optional. Leave empty to use the selected Gradio prompt preset.',
          )}
        </>
      );
    }

    if (selectedNode.type === 'voiceInput') {
      return (
        <>
          {renderTextareaField(
            'Transcript',
            'transcript',
            'Spoken instruction for the VLM node.',
          )}
          {renderPathField('Audio File Path', 'audioPath', 'Optional local audio file', ['file'])}
          {renderCheckboxField('Auto-detect intent while running', 'autoDetectIntent')}
          {renderSelectField('Intent', 'intent', VOICE_INTENTS)}
          {renderSelectField('Prompt Key', 'promptKey', VOICE_PROMPT_KEYS)}
        </>
      );
    }

    if (selectedNode.type === 'labelSuggester') {
      const apiPrompts = Array.isArray(selectedData.availablePrompts)
        ? (selectedData.availablePrompts as Array<{ key?: string }>)
            .map((prompt) => prompt.key)
            .filter((key): key is string => Boolean(key))
        : [];

      return (
        <>
          {renderSelectField('Model', 'model', VLM_MODELS)}
          {renderSelectField(
            'Prompt Preset',
            'promptKey',
            apiPrompts.length > 0 ? apiPrompts : FALLBACK_LABEL_PROMPTS,
          )}
          {renderSelectField('Modality', 'modality', VLM_MODALITIES)}
          {renderSelectField('View Plane', 'view', ['axial', 'coronal', 'sagittal'])}
          {renderNumberField('Slice Index', 'sliceIndex')}
          {renderNumberField('Max Tokens', 'maxTokens')}
          {renderNumberField('Max Labels', 'maxLabels')}
          {renderCheckboxField('Use annotation overlays when available', 'useOverlay')}
          {renderTextareaField(
            'Custom Prompt',
            'customPrompt',
            'Optional label-specific prompt.',
          )}
        </>
      );
    }

    if (selectedNode.type === 'campaignSetup') {
      return (
        <>
          {renderTextField('Campaign Name', 'campaignName', 'Brain MRI Expert Review')}
          {renderPathField('Dataset Root', 'datasetPath', 'Dataset folder with patient subfolders', ['directory'])}
          {renderTextareaField('Description', 'description', 'Optional campaign notes.')}
        </>
      );
    }

    if (selectedNode.type === 'patientAssign') {
      return (
        <>
          {renderTextField('Campaign Name', 'campaignName', 'Auto-filled from Campaign Setup')}
          {renderTextField('Expert ID', 'expertId', 'Expert user id')}
          {renderSelectField('Assignment Mode', 'assignmentMode', ASSIGNMENT_MODES)}
          {selectedData.assignmentMode === 'selected'
            ? renderTextareaField('Patient IDs', 'patientIdsText', 'Comma or newline-separated patient IDs')
            : null}
        </>
      );
    }

    if (selectedNode.type === 'campaignStatus') {
      return (
        <>
          {renderTextField('Campaign Name', 'campaignName', 'Auto-filled from campaign branch')}
        </>
      );
    }

    return (
      <div style={helpTextStyle}>
        No editable configuration is registered for this node type yet.
      </div>
    );
  };

  if (collapsed) {
    return (
      <aside style={panelShellStyle}>
        <div
          style={{
            padding: 8,
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          {renderCollapseButton('Expand inspector')}
        </div>
        <div style={collapsedBodyStyle}>
          <div
            style={{
              transform: 'rotate(90deg)',
              whiteSpace: 'nowrap',
              color: 'var(--text-secondary)',
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: 1,
              textTransform: 'uppercase',
            }}
          >
            Inspector
          </div>
        </div>
      </aside>
    );
  }

  if (selectedEdge) {
    const sourceNode = nodes.find((node) => node.id === selectedEdge.source);
    const targetNode = nodes.find((node) => node.id === selectedEdge.target);
    const edgeData = selectedEdge.data as {
      kind?: WorkflowPortKind;
      label?: string;
      color?: string;
    } | undefined;
    const portDefinition = getPortDefinition(edgeData?.kind);
    const edgeColor = edgeData?.color || portDefinition?.color || 'var(--accent-blue)';
    const edgeLabel = edgeData?.label || portDefinition?.edgeLabel || 'data';

    return (
      <aside style={panelShellStyle}>
        <div style={headerStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 18 }}>{'->'}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
                Connection
              </div>
              <div style={{ marginTop: 5 }}>
                {renderNodeChip(edgeLabel, edgeColor)}
              </div>
            </div>
            {renderCollapseButton('Collapse inspector')}
          </div>
        </div>

        <div style={bodyStyle}>
          <div style={connectionSectionStyle}>
            <div
              style={{
                ...sectionTitleStyle,
                color: 'var(--accent-blue)',
                marginBottom: 10,
              }}
            >
              Route
            </div>
            {renderConnectionBox(
              'From',
              [{ id: selectedEdge.source, node: sourceNode }],
              'var(--accent-blue)',
            )}
            {renderConnectionBox(
              'To',
              [{ id: selectedEdge.target, node: targetNode }],
              'var(--accent-green)',
            )}
          </div>

          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>Guidance</div>
            <div style={helpTextStyle}>
              This connection passes {edgeLabel} data from the source node to the target node.
              Select either endpoint to configure the node, or remove this connection below.
            </div>
          </div>

          <button
            onClick={() => {
              removeEdge(selectedEdge.id);
              setSelectedEdgeId(null);
            }}
            style={{
              width: '100%',
              padding: '8px 10px',
              borderRadius: 6,
              border: '1px solid var(--accent-red)',
              background: 'rgba(224, 92, 92, 0.1)',
              color: 'var(--accent-red)',
              fontWeight: 700,
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            Remove Connection
          </button>
        </div>
      </aside>
    );
  }

  if (!selectedNode || !selectedData) {
    return (
      <aside style={panelShellStyle}>
        <div style={headerStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
                Inspector
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                Select a node to configure it
              </div>
            </div>
            {renderCollapseButton('Collapse inspector')}
          </div>
        </div>
        <div style={bodyStyle}>
          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>Recommended first path</div>
            <div style={helpTextStyle}>
              Data Loader {'->'} Interactive Annotator, plus Segmentation Profile {'->'} Interactive Annotator for SAM2.
              Add Slice Viewer separately when you want a read-only preview branch.
            </div>
          </div>
          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>Graph</div>
            <div style={connectionRowStyle}>
              <span>Nodes</span>
              <strong>{nodes.length}</strong>
            </div>
            <div style={connectionRowStyle}>
              <span>Connections</span>
              <strong>{edges.length}</strong>
            </div>
          </div>
          <div style={helpTextStyle}>
            Use the inspector for configuration details and connection guidance as the
            node library grows.
          </div>
        </div>
      </aside>
    );
  }

  const paletteItem = nodePaletteItems.find((item) => item.type === selectedNode.type);
  const status = String(selectedData.status || 'idle');
  const allowedTargets = getAllowedTargets(selectedNode.type || '');
  const allowedSources = getAllowedSources(selectedNode.type || '');

  const incomingNodes = incoming.map((edge) => ({
    id: edge.id,
    node: nodes.find((node) => node.id === edge.source),
  }));

  const outgoingNodes = outgoing.map((edge) => ({
    id: edge.id,
    node: nodes.find((node) => node.id === edge.target),
  }));

  return (
    <aside style={panelShellStyle}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 18 }}>{paletteItem?.icon || '*'}</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
              {getNodeTitle(selectedNode)}
            </div>
          </div>
          <span style={pillStyle(statusColor[status] || 'var(--text-muted)')}>
            {status}
          </span>
          {renderCollapseButton('Collapse inspector')}
        </div>
      </div>

      <div style={bodyStyle}>
        <div style={sectionStyle}>
          <div style={sectionTitleStyle}>Configuration</div>
          {renderConfig()}
        </div>

        {selectedData.error ? (
          <div style={sectionStyle}>
            <div style={{ ...sectionTitleStyle, color: 'var(--accent-red)' }}>Last Error</div>
            <div
              style={{
                color: 'var(--accent-red)',
                background: 'rgba(224, 92, 92, 0.08)',
                border: '1px solid rgba(224, 92, 92, 0.25)',
                borderRadius: 6,
                padding: 8,
                fontSize: 11,
                lineHeight: 1.4,
                wordBreak: 'break-word',
              }}
            >
              {String(selectedData.error)}
            </div>
          </div>
        ) : null}

        <div style={connectionSectionStyle}>
          <div
            style={{
              ...sectionTitleStyle,
              color: 'var(--accent-blue)',
              marginBottom: 10,
            }}
          >
            Connections
          </div>
          {renderConnectionBox('Input from', incomingNodes, 'var(--accent-blue)')}
          {renderConnectionBox('Output to', outgoingNodes, 'var(--accent-green)')}
          {incomingNodes.length > 0 || outgoingNodes.length > 0 ? (
            <div style={dividerStyle} />
          ) : null}
          {renderCompatibilityBox(
            'Can receive from',
            allowedSources,
            'No upstream node types are valid for this node.',
            'var(--accent-blue)',
          )}
          {renderCompatibilityBox(
            'Can connect to',
            allowedTargets,
            'This is a terminal node.',
            'var(--accent-green)',
          )}
        </div>

        <div style={sectionStyle}>
          <div style={sectionTitleStyle}>Guidance</div>
          <div style={helpTextStyle}>
            {paletteItem?.description ||
              'Configure this node, connect compatible upstream data, then run the workflow.'}
          </div>
        </div>

        <button
          onClick={() => {
            removeNode(selectedNode.id);
            setSelectedNodeId(null);
          }}
          style={{
            width: '100%',
            padding: '8px 10px',
            borderRadius: 6,
            border: '1px solid var(--accent-red)',
            background: 'rgba(224, 92, 92, 0.1)',
            color: 'var(--accent-red)',
            fontWeight: 700,
            fontSize: 12,
            cursor: 'pointer',
          }}
        >
          Remove Node
        </button>
      </div>
    </aside>
  );
}
