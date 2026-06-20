/**
 * App.tsx — Main SegPro-Med Workflow Engine
 * ==========================================
 * React Flow canvas with node palette sidebar,
 * toolbar for workflow execution, and drag-and-drop node creation.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { createPortal } from 'react-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
  type Viewport,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import useWorkflowStore, { generateNodeId } from './store/workflowStore';
import { nodePaletteItems, nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import NodePalette from './panels/NodePalette';
import {
  executeWorkflow,
  isWorkflowExecutionCancelledError,
  type WorkflowRunMode,
} from './engine/executor';
import { createIsValidConnection, getAllowedSources, getAllowedTargets } from './engine/compatibility';
import { getConnectionEdgeData, getNodeContract } from './engine/nodeContracts';
import NodeInspector from './panels/NodeInspector';
import WorkflowValidationPanel from './panels/WorkflowValidationPanel';
import WorkflowTemplatePanel from './panels/WorkflowTemplatePanel';
import NodeSuggestionMenu from './components/NodeSuggestionMenu';
import NodeContextMenu from './components/NodeContextMenu';
import CrowdsourcingLoginModal from './components/CrowdsourcingLoginModal';
import CanvasEmptyState from './components/CanvasEmptyState';
import ResultModal from './components/ResultModal';
import TaskTutorialOverlay from './components/TaskTutorialOverlay';
import * as api from './api/client';
import { validateWorkflow, type WorkflowIssue } from './engine/workflowValidation';
import {
  WORKFLOW_STORAGE_KEY,
  createWorkflowSnapshot,
  parseWorkflowSnapshot,
  serializeWorkflowSnapshot,
} from './engine/workflowPersistence';
import {
  instantiateWorkflowTemplate,
  workflowTemplates,
  type WorkflowTemplate,
} from './engine/workflowTemplates';
import {
  MEDICAL_REPORT_OUTPUT_TUTORIAL,
  VLM_LABEL_SUGGESTIONS_OUTPUT_TUTORIAL,
  taskTutorials,
  type TaskTutorialConfig,
} from './engine/taskTutorials';
import {
  MEDICAL_REPORT_TEMPLATE_ID,
  TASK_DATA_AUTO_LOAD_EVENT,
  VLM_LABEL_SUGGESTIONS_TEMPLATE_ID,
  supportsTaskDataAutoLoad,
} from './engine/taskEvents';
import type {
  AnnotationShape,
  BaseNodeData,
  CampaignInfo,
  CrowdsourcingTaskItem,
  SliceAnnotationsMap,
} from './types/nodes';

interface SuggestionMenuState {
  sourceNodeId: string;
  sourceNodeType: string;
  direction: 'input' | 'output';
}

interface ClipboardFragment {
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
}

interface NodeContextMenuState {
  nodeId: string;
  nodeType?: string;
  label: string;
  icon: string;
  category?: string;
  hasDownstream: boolean;
  position: { x: number; y: number };
}

interface ActiveWorkflowRun {
  mode: WorkflowRunMode;
  nodeIds: string[];
  selectedNodeId?: string;
  startedAt: string;
}

interface ExecutionHistoryRecord {
  id: string;
  mode: WorkflowRunMode;
  status: 'success' | 'error' | 'cancelled';
  startedAt: string;
  finishedAt: string;
  nodeIds: string[];
  executedNodeIds: string[];
  failedNodeIds: string[];
  skippedNodeIds: string[];
  selectedNodeId?: string;
  message?: string;
}

interface CrowdsourcingSession {
  userId: string;
  role: string;
  expertScore: number;
  remainingTasks: api.CrowdsourcingTask[];
  assignedTasks: api.CrowdsourcingTask[];
}

const SUGGESTION_MENU_WIDTH = 226;
const SUGGESTION_MENU_GAP = 28;
const NODE_CONTEXT_MENU_WIDTH = 258;
const NODE_CONTEXT_MENU_ESTIMATED_HEIGHT = 330;
const EXECUTION_HISTORY_STORAGE_KEY = 'segpro-med.workflow.executionHistory';
const EXECUTION_HISTORY_LIMIT = 50;
const CROWDSOURCING_GUIDE_DISMISSED_KEY = 'segpro-med.workflow.crowdsourcingGuideDismissed';
const CROWDSOURCING_HELPER_HIDDEN_KEY = 'segpro-med.workflow.crowdsourcingHelperHidden';
const DEIDENTIFICATION_HELPER_HIDDEN_KEY = 'segpro-med.workflow.deidentificationHelperHidden';
const DEIDENTIFICATION_GUIDE_DISMISSED_KEY = 'segpro-med.workflow.deidentificationGuideDismissed';
const DEIDENTIFICATION_RESULT_DISMISSED_KEY = 'segpro-med.workflow.deidentificationResultDismissed';
function cloneJson<T>(value: T): T {
  if (value === undefined) return value;
  return JSON.parse(JSON.stringify(value)) as T;
}

function getEditableWorkflowFingerprint(
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
) {
  const snapshot = createWorkflowSnapshot(nodes, edges);
  return JSON.stringify({
    nodes: snapshot.nodes,
    edges: snapshot.edges,
  });
}

function loadExecutionHistory(): ExecutionHistoryRecord[] {
  if (typeof window === 'undefined') return [];

  try {
    const raw = window.localStorage.getItem(EXECUTION_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.slice(0, EXECUTION_HISTORY_LIMIT) as ExecutionHistoryRecord[]
      : [];
  } catch {
    return [];
  }
}

function getCrowdsourcingTaskLoadPath(task: api.CrowdsourcingTask) {
  return task.load_path || task.patient_path || (
    task.dataset_path && task.patient_id
      ? `${task.dataset_path}\\${task.patient_id}`
      : ''
  );
}

function taskKey(task: api.CrowdsourcingTask) {
  return `${task.campaign_id}:${task.patient_id}`;
}

function buildCrowdsourcingTaskItems(
  assignedTasks: api.CrowdsourcingTask[],
  remainingTasks: api.CrowdsourcingTask[],
  currentTask: api.CrowdsourcingTask,
): CrowdsourcingTaskItem[] {
  const remainingKeys = new Set(remainingTasks.map(taskKey));
  const currentKey = taskKey(currentTask);
  const sourceTasks = assignedTasks.length > 0 ? assignedTasks : remainingTasks;
  const taskMap = new Map<string, api.CrowdsourcingTask>();

  for (const task of sourceTasks) taskMap.set(taskKey(task), task);
  taskMap.set(currentKey, currentTask);

  return [...taskMap.values()].map((task) => {
    const key = taskKey(task);
    return {
      campaignId: task.campaign_id,
      patientId: task.patient_id,
      modality: task.modality,
      loadPath: getCrowdsourcingTaskLoadPath(task),
      status: key === currentKey
        ? 'current'
        : remainingKeys.has(key)
          ? 'pending'
          : 'completed',
    };
  });
}

function mapWorkflowCampaign(campaign: api.CollaborationCampaign): CampaignInfo {
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

function shapeToSubmissionRecord(
  shape: AnnotationShape,
  sliceIndex: number,
  view: string,
  index: number,
) {
  const label = shape.label || 'Annotation';
  const base = {
    annotation_id: `workflow_${sliceIndex}_${view}_${index}`,
    slice_index: sliceIndex,
    view_type: view,
    type: shape.type,
    label,
    color: shape.color,
    annotated_by: shape.annotatedBy,
    annotated_at: shape.annotatedAt,
    updated_by: shape.updatedBy,
    updated_at: shape.updatedAt,
  };

  if (shape.type === 'rect') {
    const bbox = [shape.x, shape.y, shape.x + shape.width, shape.y + shape.height];
    return {
      ...base,
      bbox,
      coordinates: [
        [shape.x, shape.y],
        [shape.x + shape.width, shape.y],
        [shape.x + shape.width, shape.y + shape.height],
        [shape.x, shape.y + shape.height],
      ],
    };
  }

  if (shape.type === 'circle') {
    return {
      ...base,
      bbox: [
        shape.centerX - shape.radius,
        shape.centerY - shape.radius,
        shape.centerX + shape.radius,
        shape.centerY + shape.radius,
      ],
      coordinates: [[shape.centerX, shape.centerY]],
    };
  }

  if (shape.type === 'point') {
    return {
      ...base,
      coordinates: [[shape.x, shape.y]],
      points: [{ x: shape.x, y: shape.y }],
    };
  }

  return {
    ...base,
    coordinates: shape.points.map((point) => [point.x, point.y]),
    points: shape.points,
  };
}

function buildCrowdsourcingAnnotationData(
  nodes: Node<BaseNodeData>[],
  currentTask: api.CrowdsourcingTask,
  userId: string,
) {
  const annotator = nodes.find((node) => node.type === 'interactiveAnnotator');
  const data = (annotator?.data || {}) as Record<string, unknown>;
  const view = typeof data.view === 'string' ? data.view : 'axial';
  const currentSliceIndex = Number(data.sliceIndex ?? 0);
  const currentAnnotations = Array.isArray(data.annotations)
    ? data.annotations as AnnotationShape[]
    : [];
  const sliceAnnotationsMap = (data.sliceAnnotationsMap || {}) as SliceAnnotationsMap;
  const records: Array<Record<string, unknown>> = [];
  const labels = new Map<string, { label: string; slice_index: number; view_type: string; color?: string }>();

  const addShape = (shape: AnnotationShape, sliceIndex: number, index: number) => {
    records.push(shapeToSubmissionRecord(shape, sliceIndex, view, index));
    if (shape.label) {
      labels.set(`${shape.label}:${sliceIndex}:${view}`, {
        label: shape.label,
        slice_index: sliceIndex,
        view_type: view,
        color: shape.color,
      });
    }
  };

  Object.entries(sliceAnnotationsMap).forEach(([sliceKey, shapes]) => {
    const sliceIndex = Number(sliceKey);
    (shapes || []).forEach((shape, index) => addShape(shape, sliceIndex, index));
  });

  if (records.length === 0) {
    currentAnnotations.forEach((shape, index) => addShape(shape, currentSliceIndex, index));
  }

  return {
    campaign_name: currentTask.campaign_id,
    expert_id: userId,
    patient_id: currentTask.patient_id,
    source_path: getCrowdsourcingTaskLoadPath(currentTask),
    annotations: records,
    labels: [...labels.values()],
    total_annotations: records.length,
  };
}

function getRunModeLabel(mode: WorkflowRunMode) {
  if (mode === 'selected') return 'selected node';
  if (mode === 'downstream') return 'downstream branch';
  return 'workflow';
}

function getExecutionScopeNodeIds(
  mode: WorkflowRunMode,
  selectedNodeId: string | undefined,
  nodes: Node<BaseNodeData>[],
  edges: Edge[],
) {
  if (mode === 'all') {
    return new Set(nodes.map((node) => node.id));
  }

  if (!selectedNodeId) {
    return new Set<string>();
  }

  if (mode === 'selected') {
    return new Set([selectedNodeId]);
  }

  const scope = new Set<string>([selectedNodeId]);
  const queue = [selectedNodeId];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) continue;

    for (const edge of edges) {
      if (edge.source !== current || scope.has(edge.target)) continue;
      scope.add(edge.target);
      queue.push(edge.target);
    }
  }

  return scope;
}

function getNodeContextMenuPosition(clientX: number, clientY: number) {
  const margin = 8;
  const maxX = window.innerWidth - NODE_CONTEXT_MENU_WIDTH - margin;
  const maxY = window.innerHeight - NODE_CONTEXT_MENU_ESTIMATED_HEIGHT - margin;

  return {
    x: Math.max(margin, Math.min(clientX, maxX)),
    y: Math.max(margin, Math.min(clientY, maxY)),
  };
}

const appStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  display: 'flex',
};

const canvasContainerStyle: React.CSSProperties = {
  flex: 1,
  height: '100%',
  position: 'relative',
};

const toolbarStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  right: 12,
  zIndex: 10,
  display: 'flex',
  gap: 8,
  alignItems: 'center',
};

const unsavedBadgeStyle: React.CSSProperties = {
  padding: '7px 10px',
  borderRadius: 6,
  border: '1px solid color-mix(in srgb, var(--accent-orange) 52%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-orange) 14%, var(--bg-secondary))',
  color: 'var(--accent-orange)',
  fontSize: 11,
  fontWeight: 800,
  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.18)',
  whiteSpace: 'nowrap',
};

const crowdsourcingBannerBaseStyle: React.CSSProperties = {
  position: 'absolute',
  left: 74,
  bottom: 12,
  zIndex: 10,
  width: 342,
  maxWidth: 'calc(100% - 92px)',
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--accent-green) 35%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-green) 9%, var(--bg-secondary))',
  boxShadow: 'var(--shadow)',
  padding: 10,
  transformOrigin: 'bottom left',
};

const crowdsourcingMinimizedBaseStyle: React.CSSProperties = {
  position: 'absolute',
  left: 74,
  bottom: 12,
  zIndex: 10,
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--accent-green) 35%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-green) 9%, var(--bg-secondary))',
  boxShadow: 'var(--shadow)',
  padding: '6px 8px',
  display: 'flex',
  gap: 8,
  alignItems: 'center',
  transformOrigin: 'bottom left',
};

const crowdsourcingMetaStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  color: 'var(--text-muted)',
  fontSize: 11,
  marginTop: 6,
};

const crowdsourcingHelperStyle: React.CSSProperties = {
  position: 'absolute',
  top: 62,
  left: 12,
  right: 12,
  zIndex: 9,
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--accent-blue) 35%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-blue) 8%, var(--bg-secondary))',
  boxShadow: 'var(--shadow)',
  padding: '10px 12px',
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  pointerEvents: 'none',
};

const guideOverlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 9999,
  background: 'rgba(0, 0, 0, 0.6)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 16,
  backdropFilter: 'blur(4px)',
};

const guideModalStyle: React.CSSProperties = {
  width: 460,
  maxWidth: '100%',
  maxHeight: '82vh',
  overflow: 'hidden',
  borderRadius: 12,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
  display: 'flex',
  flexDirection: 'column',
};

const confirmModalStyle: React.CSSProperties = {
  width: 420,
  maxWidth: '100%',
  borderRadius: 10,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: '0 18px 52px rgba(0, 0, 0, 0.48)',
  overflow: 'hidden',
};

const btnStyle = (color: string, disabled = false, active = false): React.CSSProperties => ({
  padding: '8px 14px',
  borderRadius: 6,
  border: `1px solid color-mix(in srgb, ${color} ${active ? '72%' : '54%'}, var(--border-color))`,
  background: disabled
    ? 'var(--bg-secondary)'
    : `color-mix(in srgb, ${color} ${active ? '25%' : '15%'}, var(--bg-secondary))`,
  color,
  fontWeight: 800,
  fontSize: 12,
  cursor: disabled ? 'not-allowed' : 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 6,
  minHeight: 34,
  boxShadow: disabled ? 'none' : '0 2px 10px rgba(0, 0, 0, 0.18)',
  opacity: disabled ? 0.45 : 1,
  transition: 'background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease',
});

const iconButtonStyle = (color: string, disabled = false): React.CSSProperties => ({
  ...btnStyle(color, disabled),
  width: 34,
  minWidth: 34,
  padding: 0,
  fontSize: 16,
  lineHeight: 1,
});

const compactButtonStyle = (color: string, disabled = false): React.CSSProperties => ({
  ...btnStyle(color, disabled),
  padding: '5px 9px',
  minHeight: 28,
  fontSize: 11,
});

const compactIconButtonStyle = (color: string, disabled = false): React.CSSProperties => ({
  ...compactButtonStyle(color, disabled),
  width: 28,
  minWidth: 28,
  padding: 0,
  fontSize: 15,
  lineHeight: 1,
});

const miniMapToggleStyle: React.CSSProperties = {
  ...compactButtonStyle('var(--accent-blue)'),
  position: 'absolute',
  right: 12,
  bottom: 12,
  zIndex: 8,
};

const splitRunButtonStyle = (disabled = false): React.CSSProperties => ({
  ...btnStyle('var(--accent-green)', disabled),
  borderTopRightRadius: 0,
  borderBottomRightRadius: 0,
  borderRight: 'none',
  paddingRight: 12,
});

const splitArrowButtonStyle = (active = false): React.CSSProperties => ({
  ...btnStyle('var(--accent-green)', false, active),
  borderTopLeftRadius: 0,
  borderBottomLeftRadius: 0,
  minWidth: 34,
  padding: '8px 9px',
});

const workflowMenuStyle: React.CSSProperties = {
  position: 'absolute',
  top: 40,
  right: 0,
  width: 220,
  borderRadius: 7,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
  zIndex: 20,
};

const workflowMenuItemStyle = (color: string, disabled = false): React.CSSProperties => ({
  width: '100%',
  border: 'none',
  borderBottom: '1px solid rgba(255,255,255,0.06)',
  background: 'transparent',
  color: disabled ? 'var(--text-muted)' : color,
  cursor: disabled ? 'not-allowed' : 'pointer',
  padding: '9px 10px',
  textAlign: 'left',
  fontSize: 12,
  fontWeight: 800,
  opacity: disabled ? 0.55 : 1,
});

function setButtonHover(
  element: HTMLElement,
  color: string,
  hovered: boolean,
  disabled = false,
  active = false,
) {
  if (disabled) return;
  element.style.background = hovered
    ? `color-mix(in srgb, ${color} 27%, var(--bg-secondary))`
    : `color-mix(in srgb, ${color} ${active ? '25%' : '15%'}, var(--bg-secondary))`;
  element.style.borderColor = hovered
    ? `color-mix(in srgb, ${color} 82%, var(--border-color))`
    : `color-mix(in srgb, ${color} ${active ? '72%' : '54%'}, var(--border-color))`;
  element.style.boxShadow = hovered
    ? `0 4px 14px color-mix(in srgb, ${color} 18%, rgba(0, 0, 0, 0.2))`
    : '0 2px 10px rgba(0, 0, 0, 0.18)';
}

function setMenuItemHover(element: HTMLElement, hovered: boolean, disabled = false) {
  if (disabled) return;
  element.style.background = hovered ? 'var(--bg-tertiary)' : 'transparent';
}

function isEditableShortcutTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tagName = target.tagName.toLowerCase();
  return target.isContentEditable || ['input', 'textarea', 'select'].includes(tagName);
}

const noticeStyle = (
  type: 'info' | 'success' | 'warning' | 'error',
): React.CSSProperties => {
  const colors = {
    info: 'var(--accent-blue)',
    success: 'var(--accent-green)',
    warning: 'var(--accent-orange)',
    error: 'var(--accent-red)',
  };
  const color = colors[type];

  return {
    position: 'absolute',
    top: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 11,
    maxWidth: 520,
    padding: '8px 12px',
    borderRadius: 6,
    border: `1px solid ${color}55`,
    background: `color-mix(in srgb, ${color} 14%, var(--bg-secondary))`,
    color: 'var(--text-primary)',
    fontSize: 12,
    boxShadow: 'var(--shadow)',
  };
};

export default function App() {
  const {
    nodes,
    edges,
    historyPast,
    historyFuture,
    workflowNotice,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    addNodesAndConnect,
    insertGraphFragment,
    updateNodeData,
    removeNode,
    clearWorkflow,
    replaceWorkflow,
    checkpointHistory,
    undoWorkflow,
    redoWorkflow,
    selectedNodeId,
    setSelectedNodeId,
    setSelectedEdgeId,
    setWorkflowNotice,
  } = useWorkflowStore();
  const [suggestionMenu, setSuggestionMenu] = useState<SuggestionMenuState | null>(null);
  const [showValidationPanel, setShowValidationPanel] = useState(false);
  const [showTemplatePanel, setShowTemplatePanel] = useState(false);
  const [showTaskStartModal, setShowTaskStartModal] = useState(false);
  const [showWorkflowMenu, setShowWorkflowMenu] = useState(false);
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [nodeContextMenu, setNodeContextMenu] = useState<NodeContextMenuState | null>(null);
  const [savedWorkflowFingerprint, setSavedWorkflowFingerprint] = useState(() =>
    getEditableWorkflowFingerprint([], []),
  );
  const [activeRun, setActiveRun] = useState<ActiveWorkflowRun | null>(null);
  const [executionHistory, setExecutionHistory] = useState<ExecutionHistoryRecord[]>(() =>
    loadExecutionHistory(),
  );
  const [showCrowdsourcingLogin, setShowCrowdsourcingLogin] = useState(false);
  const [crowdsourcingLoginIntent, setCrowdsourcingLoginIntent] = useState<'session' | 'crowdsourcing'>('session');
  const [showCrowdsourcingConfirm, setShowCrowdsourcingConfirm] = useState(false);
  const [crowdsourcingTaskConfirm, setCrowdsourcingTaskConfirm] = useState<'submit' | 'next' | null>(null);
  const [crowdsourcingSession, setCrowdsourcingSession] = useState<CrowdsourcingSession | null>(null);
  const [currentCrowdsourcingTask, setCurrentCrowdsourcingTask] = useState<api.CrowdsourcingTask | null>(null);
  const [crowdsourcingTaskQueue, setCrowdsourcingTaskQueue] = useState<api.CrowdsourcingTask[]>([]);
  const [crowdsourcingSubmitBusy, setCrowdsourcingSubmitBusy] = useState(false);
  const [crowdsourcingAutoRunToken, setCrowdsourcingAutoRunToken] = useState(0);
  const [crowdsourcingPanelMinimized, setCrowdsourcingPanelMinimized] = useState(false);
  const [showCrowdsourcingGuide, setShowCrowdsourcingGuide] = useState(false);
  const [doNotShowCrowdsourcingGuide, setDoNotShowCrowdsourcingGuide] = useState(false);
  const [miniMapMinimized, setMiniMapMinimized] = useState(true);
  const [viewportZoom, setViewportZoom] = useState(1);
  const [crowdsourcingHelperHidden, setCrowdsourcingHelperHidden] = useState(() => (
    typeof window !== 'undefined' &&
    window.localStorage.getItem(CROWDSOURCING_HELPER_HIDDEN_KEY) === 'true'
  ));
  const [deidentificationHelperHidden, setDeidentificationHelperHidden] = useState(() => (
    typeof window !== 'undefined' &&
    window.localStorage.getItem(DEIDENTIFICATION_HELPER_HIDDEN_KEY) === 'true'
  ));
  const [showDeidentificationGuide, setShowDeidentificationGuide] = useState(false);
  const [doNotShowDeidentificationGuide, setDoNotShowDeidentificationGuide] = useState(false);
  const [showDeidentificationResult, setShowDeidentificationResult] = useState(false);
  const [doNotShowDeidentificationResult, setDoNotShowDeidentificationResult] = useState(false);
  const [deidentificationResultPath, setDeidentificationResultPath] = useState<string | undefined>();
  const [activeTaskTutorial, setActiveTaskTutorial] = useState<TaskTutorialConfig | null>(null);
  const [doNotShowTaskTutorial, setDoNotShowTaskTutorial] = useState(false);
  const [tutorialViewportSignal, setTutorialViewportSignal] = useState(0);
  const [pendingTemplate, setPendingTemplate] = useState<WorkflowTemplate | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null);
  const workflowFileInputRef = useRef<HTMLInputElement | null>(null);
  const clipboardRef = useRef<ClipboardFragment | null>(null);
  const activeRunControllerRef = useRef<AbortController | null>(null);
  const taskDataAutoLoadRef = useRef<string | null>(null);
  const labelSuggestionGuideRef = useRef<string | null>(null);
  const canUndo = historyPast.length > 0;
  const canRedo = historyFuture.length > 0;
  const currentWorkflowFingerprint = useMemo(
    () => getEditableWorkflowFingerprint(nodes, edges),
    [nodes, edges],
  );
  const hasUnsavedChanges = nodes.length > 0 && currentWorkflowFingerprint !== savedWorkflowFingerprint;
  const crowdsourcingPanelScale = Math.max(0.72, Math.min(1.28, viewportZoom));
  const isWorkflowRunning = Boolean(activeRun);
  const hasDeidentificationWorkflow = nodes.some((node) => node.type === 'deidentifyNode');
  const selectedExecutionNodeId = selectedNodeIds[0] || selectedNodeId || undefined;
  const isAdminCrowdsourcingWorkspaceVisible = crowdsourcingSession?.role === 'admin' &&
    nodes.some((node) => ['campaignSetup', 'patientAssign', 'campaignStatus'].includes(node.type || ''));
  const currentTaskAnnotationCount = useMemo(() => {
    const annotator = nodes.find((node) => node.type === 'interactiveAnnotator');
    const annotatorData = annotator?.data as Record<string, unknown> | undefined;
    const currentAnnotations = Array.isArray(annotatorData?.annotations)
      ? annotatorData.annotations.length
      : 0;
    const sliceAnnotationsMap = annotatorData?.sliceAnnotationsMap as
      | Record<string, unknown[]>
      | undefined;

    if (!sliceAnnotationsMap) return currentAnnotations;

    const total = Object.values(sliceAnnotationsMap).reduce((sum, value) => (
      sum + (Array.isArray(value) ? value.length : 0)
    ), 0);

    return Math.max(currentAnnotations, total);
  }, [nodes]);

  // --- Connection validation (compatibility matrix) ---
  const isValidConnection = useMemo(
    () => createIsValidConnection(nodes, edges),
    [nodes, edges],
  );

  const validationIssues = useMemo(
    () => validateWorkflow(nodes, edges),
    [nodes, edges],
  );

  useEffect(() => {
    if (!workflowNotice) return;
    const timeout = window.setTimeout(() => {
      setWorkflowNotice(null);
    }, workflowNotice.type === 'error' ? 8000 : 3500);

    return () => window.clearTimeout(timeout);
  }, [workflowNotice, setWorkflowNotice]);

  useEffect(() => {
    const openMenu = (event: Event) => {
      const customEvent = event as CustomEvent<{
        sourceNodeId: string;
        sourceNodeType: string;
        direction: 'input' | 'output';
      }>;

      setSuggestionMenu({
        sourceNodeId: customEvent.detail.sourceNodeId,
        sourceNodeType: customEvent.detail.sourceNodeType,
        direction: customEvent.detail.direction,
      });
      setSelectedNodeId(customEvent.detail.sourceNodeId);
    };

    window.addEventListener('segpro:open-node-suggestions', openMenu);
    return () => {
      window.removeEventListener('segpro:open-node-suggestions', openMenu);
    };
  }, [setSelectedNodeId]);

  // --- Drag & Drop from palette ---
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const nodeType = event.dataTransfer.getData('application/reactflow-type');
      const dataStr = event.dataTransfer.getData('application/reactflow-data');

      if (!nodeType || !reactFlowInstance.current) return;

      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const defaultData = dataStr ? JSON.parse(dataStr) : { label: nodeType, status: 'idle' };

      addNode({
        id: generateNodeId(),
        type: nodeType,
        position,
        style: nodeType === 'interactiveAnnotator' ? { width: 430, height: 560 } : undefined,
        data: defaultData,
      });
    },
    [addNode],
  );

  const appendExecutionHistory = useCallback((record: ExecutionHistoryRecord) => {
    setExecutionHistory((current) => {
      const next = [record, ...current].slice(0, EXECUTION_HISTORY_LIMIT);
      try {
        window.localStorage.setItem(EXECUTION_HISTORY_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // Execution history is helpful but should never block workflow runs.
      }
      return next;
    });
  }, []);

  const handleCancelRun = useCallback(() => {
    activeRunControllerRef.current?.abort();
    setNodeContextMenu(null);
    setWorkflowNotice({
      type: 'warning',
      message: 'Cancelling workflow run...',
    });
  }, [setWorkflowNotice]);

  // --- Workflow execution ---
  const handleRun = useCallback(async (
    mode: WorkflowRunMode = 'all',
    explicitStartNodeId?: string,
  ) => {
    setShowWorkflowMenu(false);
    setNodeContextMenu(null);

    if (nodes.length === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Add at least one node before running the workflow.',
      });
      return;
    }

    if (activeRunControllerRef.current) {
      setWorkflowNotice({
        type: 'warning',
        message: 'A workflow run is already in progress.',
      });
      return;
    }

    const runNodeId = explicitStartNodeId || selectedExecutionNodeId;
    const scopedNodeIds = getExecutionScopeNodeIds(
      mode,
      runNodeId,
      nodes,
      edges,
    );
    const modeLabel = getRunModeLabel(mode);

    if (scopedNodeIds.size === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Select a node before running this action.',
      });
      return;
    }

    const scopedBlockingIssues = validationIssues.filter((issue) =>
      issue.severity === 'error' && (!issue.nodeId || scopedNodeIds.has(issue.nodeId)),
    );

    if (scopedBlockingIssues.length > 0) {
      const firstIssue = scopedBlockingIssues[0];
      if (firstIssue.nodeId) {
        setSelectedNodeId(firstIssue.nodeId);
        setSelectedNodeIds([firstIssue.nodeId]);
      }
      setShowTemplatePanel(false);
      setShowValidationPanel(true);
      setWorkflowNotice({
        type: 'warning',
        message: firstIssue.message,
      });
      return;
    }

    setShowValidationPanel(false);

    for (const node of nodes) {
      if (scopedNodeIds.has(node.id)) {
        updateNodeData(node.id, { status: 'idle', error: undefined });
      }
    }

    const controller = new AbortController();
    const startedAt = new Date().toISOString();
    const runBase = {
      id: `run_${Date.now()}`,
      mode,
      startedAt,
      nodeIds: [...scopedNodeIds],
      selectedNodeId: runNodeId,
    };

    activeRunControllerRef.current = controller;
    setActiveRun({
      mode,
      nodeIds: [...scopedNodeIds],
      selectedNodeId: runNodeId,
      startedAt,
    });

    try {
      const result = await executeWorkflow(nodes, edges, updateNodeData, {
        mode,
        startNodeId: runNodeId,
        signal: controller.signal,
      });
      appendExecutionHistory({
        ...runBase,
        status: 'success',
        finishedAt: new Date().toISOString(),
        executedNodeIds: result.executedNodeIds,
        failedNodeIds: result.failedNodeIds,
        skippedNodeIds: result.skippedNodeIds,
        message: `${modeLabel} completed.`,
      });
      setWorkflowNotice({
        type: 'success',
        message: `${modeLabel[0].toUpperCase()}${modeLabel.slice(1)} completed (${result.executedNodeIds.length} node${result.executedNodeIds.length === 1 ? '' : 's'}).`,
      });
      const deidentifyNodeId = nodes.find((node) =>
        node.type === 'deidentifyNode' && result.executedNodeIds.includes(node.id),
      )?.id;
      if (deidentifyNodeId) {
        const deidentifyResult = result.results.get(deidentifyNodeId);
        setDeidentificationResultPath(
          typeof deidentifyResult?.outputPath === 'string'
            ? deidentifyResult.outputPath
            : typeof deidentifyResult?.filePath === 'string'
              ? deidentifyResult.filePath
              : undefined,
        );
        const dismissed = typeof window !== 'undefined' &&
          window.localStorage.getItem(DEIDENTIFICATION_RESULT_DISMISSED_KEY) === 'true';
        if (!dismissed) {
          setDoNotShowDeidentificationResult(false);
          setShowDeidentificationResult(true);
        }
      }
      const medicalReportNode = nodes.find((node) =>
        node.type === 'medgemmaNode' &&
        node.data?.taskTemplateId === MEDICAL_REPORT_TEMPLATE_ID &&
        node.data?.taskNodeKey === 'medgemma' &&
        result.executedNodeIds.includes(node.id) &&
        Boolean(result.results.get(node.id)?.vlmResult),
      );
      if (medicalReportNode) {
        const dismissed = typeof window !== 'undefined' &&
          window.localStorage.getItem(MEDICAL_REPORT_OUTPUT_TUTORIAL.storageKey) === 'true';
        if (!dismissed) {
          setDoNotShowTaskTutorial(false);
          window.setTimeout(() => {
            reactFlowInstance.current?.fitView?.({
              nodes: [{ id: medicalReportNode.id }],
              padding: 0.35,
              duration: 320,
            });
          }, 80);
          window.setTimeout(() => {
            setTutorialViewportSignal((value) => value + 1);
            setActiveTaskTutorial(MEDICAL_REPORT_OUTPUT_TUTORIAL);
          }, 460);
        }
      }
      const labelSuggestionNode = nodes.find((node) => {
        if (
          node.type !== 'labelSuggester' ||
          node.data?.taskTemplateId !== 'vlm-label-suggestions' ||
          node.data?.taskNodeKey !== 'labels' ||
          !result.executedNodeIds.includes(node.id)
        ) {
          return false;
        }

        const labels = result.results.get(node.id)?.labelSuggestions;
        return Array.isArray(labels) && labels.length > 0;
      });
      if (labelSuggestionNode) {
        const labelResult = result.results.get(labelSuggestionNode.id);
        const labels = Array.isArray(labelResult?.labelSuggestions)
          ? labelResult.labelSuggestions.map((label) => String(label)).filter(Boolean)
          : [];
        const sliceIndex = Number(labelResult?.sliceIndex ?? labelSuggestionNode.data?.sliceIndex ?? 0);
        const view = String(labelResult?.view || labelSuggestionNode.data?.view || 'axial');
        const guideKey = `${labelSuggestionNode.id}:${view}:${sliceIndex}:${labels.join('|')}`;
        const dismissed = typeof window !== 'undefined' &&
          window.localStorage.getItem(VLM_LABEL_SUGGESTIONS_OUTPUT_TUTORIAL.storageKey) === 'true';

        if (!dismissed && labelSuggestionGuideRef.current !== guideKey) {
          labelSuggestionGuideRef.current = guideKey;
          setDoNotShowTaskTutorial(false);
          window.setTimeout(() => {
            reactFlowInstance.current?.fitView?.({
              nodes: [{ id: labelSuggestionNode.id }],
              padding: 0.35,
              duration: 320,
            });
          }, 80);
          window.setTimeout(() => {
            setTutorialViewportSignal((value) => value + 1);
            setActiveTaskTutorial(VLM_LABEL_SUGGESTIONS_OUTPUT_TUTORIAL);
          }, 460);
        }
      }
    } catch (err) {
      const finishedAt = new Date().toISOString();
      const isCancelled = isWorkflowExecutionCancelledError(err) || controller.signal.aborted;
      const executedNodeIds = err instanceof Error && 'executedNodeIds' in err
        ? (err.executedNodeIds as string[])
        : [];
      const failedNodeIds = err instanceof Error && 'failedNodeIds' in err
        ? (err.failedNodeIds as string[])
        : [];
      const skippedNodeIds = err instanceof Error && 'skippedNodeIds' in err
        ? (err.skippedNodeIds as string[])
        : [];
      const message = isCancelled
        ? 'Workflow run cancelled.'
        : err instanceof Error
          ? err.message
          : String(err);

      appendExecutionHistory({
        ...runBase,
        status: isCancelled ? 'cancelled' : 'error',
        finishedAt,
        executedNodeIds,
        failedNodeIds,
        skippedNodeIds,
        message,
      });

      setWorkflowNotice({
        type: isCancelled ? 'warning' : 'error',
        message,
      });
    } finally {
      if (activeRunControllerRef.current === controller) {
        activeRunControllerRef.current = null;
        setActiveRun(null);
      }
    }
  }, [
    appendExecutionHistory,
    edges,
    nodes,
    selectedExecutionNodeId,
    setSelectedNodeId,
    setWorkflowNotice,
    updateNodeData,
    validationIssues,
  ]);

  const runTaskDataAutoLoad = useCallback(async (
    loaderNodeId: string,
    templateId: string | undefined,
    requestedPath?: string,
  ) => {
    const workflowState = useWorkflowStore.getState();
    const loaderNode = workflowState.nodes.find((node) => node.id === loaderNodeId);
    const loaderData = loaderNode?.data as Record<string, unknown> | undefined;
    const taskTemplateId = typeof loaderData?.taskTemplateId === 'string'
      ? loaderData.taskTemplateId
      : templateId;

    if (!loaderNode || loaderNode.type !== 'dataLoader') return;
    if (!supportsTaskDataAutoLoad(taskTemplateId)) return;
    if (loaderData?.taskNodeKey !== 'loader') return;

    const taskTitle = taskTemplateId === MEDICAL_REPORT_TEMPLATE_ID
      ? 'Medical Report Generation'
      : taskTemplateId === VLM_LABEL_SUGGESTIONS_TEMPLATE_ID
        ? 'VLM Label Suggestions'
        : 'AI Segmentation Review';
    const successMessage = taskTemplateId === MEDICAL_REPORT_TEMPLATE_ID
      ? 'Image loaded in Interactive Annotator. Review the slice, then create the medical report when ready.'
      : taskTemplateId === VLM_LABEL_SUGGESTIONS_TEMPLATE_ID
        ? 'Image loaded in Interactive Annotator. Review the slice, then suggest labels when ready.'
        : 'Image loaded in Interactive Annotator. Review the slice, then run Batch SAM2 when ready.';

    const currentPath = typeof loaderData.path === 'string' ? loaderData.path.trim() : '';
    if (!currentPath) {
      setWorkflowNotice({
        type: 'warning',
        message: `Choose a valid file or folder before loading the ${taskTitle} task.`,
      });
      return;
    }
    if (requestedPath && requestedPath.trim() !== currentPath) return;

    const annotatorNode = workflowState.nodes.find((node) =>
      node.data?.taskTemplateId === taskTemplateId &&
      node.data?.taskNodeKey === 'annotator' &&
      node.type === 'interactiveAnnotator',
    );
    if (!annotatorNode) return;

    const runKey = `${taskTemplateId}:${loaderNodeId}:${currentPath}`;
    if (taskDataAutoLoadRef.current === runKey) return;
    if (activeRunControllerRef.current) {
      setWorkflowNotice({
        type: 'warning',
        message: 'A workflow run is already in progress. The selected data will not auto-load until the current run finishes.',
      });
      return;
    }

    taskDataAutoLoadRef.current = runKey;
    const controller = new AbortController();
    const startedAt = new Date().toISOString();
    const scopedNodeIds = [loaderNode.id, annotatorNode.id];

    activeRunControllerRef.current = controller;
    setActiveRun({
      mode: 'selected',
      nodeIds: scopedNodeIds,
      selectedNodeId: loaderNode.id,
      startedAt,
    });
    setWorkflowNotice({
      type: 'info',
      message: `Loading selected image data for ${taskTitle}...`,
    });

    try {
      for (const nodeId of scopedNodeIds) {
        updateNodeData(nodeId, { status: 'idle', error: undefined });
      }

      const loaderResult = await executeWorkflow(
        workflowState.nodes,
        workflowState.edges,
        updateNodeData,
        {
          mode: 'selected',
          startNodeId: loaderNode.id,
          signal: controller.signal,
        },
      );
      const afterLoaderState = useWorkflowStore.getState();
      const refreshedAnnotator = afterLoaderState.nodes.find((node) =>
        node.data?.taskTemplateId === taskTemplateId &&
        node.data?.taskNodeKey === 'annotator' &&
        node.type === 'interactiveAnnotator',
      );

      if (!refreshedAnnotator) {
        throw new Error(`Interactive Annotator is not available in this ${taskTitle} task.`);
      }

      const annotatorResult = await executeWorkflow(
        afterLoaderState.nodes,
        afterLoaderState.edges,
        updateNodeData,
        {
          mode: 'selected',
          startNodeId: refreshedAnnotator.id,
          signal: controller.signal,
        },
      );
      const executedNodeIds = [
        ...loaderResult.executedNodeIds,
        ...annotatorResult.executedNodeIds.filter((nodeId) =>
          !loaderResult.executedNodeIds.includes(nodeId),
        ),
      ];

      appendExecutionHistory({
        id: `run_${Date.now()}`,
        mode: 'selected',
        status: 'success',
        startedAt,
        finishedAt: new Date().toISOString(),
        nodeIds: scopedNodeIds,
        selectedNodeId: loaderNode.id,
        executedNodeIds,
        failedNodeIds: [],
        skippedNodeIds: [],
        message: `${taskTitle} data loaded.`,
      });
      setWorkflowNotice({
        type: 'success',
        message: successMessage,
      });
    } catch (err) {
      const isCancelled = isWorkflowExecutionCancelledError(err) || controller.signal.aborted;
      const failedNodeIds = err instanceof Error && 'failedNodeIds' in err
        ? (err.failedNodeIds as string[])
        : [];
      const skippedNodeIds = err instanceof Error && 'skippedNodeIds' in err
        ? (err.skippedNodeIds as string[])
        : [];
      const executedNodeIds = err instanceof Error && 'executedNodeIds' in err
        ? (err.executedNodeIds as string[])
        : [];
      const message = isCancelled
        ? `${taskTitle} auto-load cancelled.`
        : err instanceof Error
          ? err.message
          : 'Could not load the selected image data.';

      appendExecutionHistory({
        id: `run_${Date.now()}`,
        mode: 'selected',
        status: isCancelled ? 'cancelled' : 'error',
        startedAt,
        finishedAt: new Date().toISOString(),
        nodeIds: scopedNodeIds,
        selectedNodeId: loaderNode.id,
        executedNodeIds,
        failedNodeIds,
        skippedNodeIds,
        message,
      });
      setWorkflowNotice({
        type: isCancelled ? 'warning' : 'error',
        message,
      });
    } finally {
      if (activeRunControllerRef.current === controller) {
        activeRunControllerRef.current = null;
        setActiveRun(null);
      }
      if (taskDataAutoLoadRef.current === runKey) {
        taskDataAutoLoadRef.current = null;
      }
    }
  }, [
    appendExecutionHistory,
    setWorkflowNotice,
    updateNodeData,
  ]);

  useEffect(() => {
    const handleTaskDataAutoLoad = (event: Event) => {
      const detail = (event as CustomEvent<{
        nodeId?: string;
        templateId?: string;
        path?: string;
      }>).detail;
      if (!detail?.nodeId) return;

      window.setTimeout(() => {
        void runTaskDataAutoLoad(detail.nodeId as string, detail.templateId, detail.path);
      }, 0);
    };

    window.addEventListener(TASK_DATA_AUTO_LOAD_EVENT, handleTaskDataAutoLoad);
    return () => {
      window.removeEventListener(TASK_DATA_AUTO_LOAD_EVENT, handleTaskDataAutoLoad);
    };
  }, [runTaskDataAutoLoad]);

  useEffect(() => {
    const handleNodeRunRequest = (event: Event) => {
      const detail = (event as CustomEvent<{ nodeId?: string }>).detail;
      if (!detail?.nodeId) return;
      void handleRun('selected', detail.nodeId);
    };

    window.addEventListener('segpro:run-node', handleNodeRunRequest);
    return () => {
      window.removeEventListener('segpro:run-node', handleNodeRunRequest);
    };
  }, [handleRun]);

  const handleSaveWorkflow = useCallback(() => {
    setShowWorkflowMenu(false);

    if (nodes.length === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Add at least one node before saving the workflow.',
      });
      return;
    }

    const snapshot = createWorkflowSnapshot(nodes, edges);
    const serialized = serializeWorkflowSnapshot(snapshot);

    try {
      window.localStorage.setItem(WORKFLOW_STORAGE_KEY, serialized);
    } catch {
      // File export still works when browser storage is unavailable.
    }
    setSavedWorkflowFingerprint(getEditableWorkflowFingerprint(snapshot.nodes, snapshot.edges));

    const blob = new Blob([serialized], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `segpro-workflow-${snapshot.exportedAt.replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    setWorkflowNotice({
      type: 'success',
      message: 'Workflow saved as JSON.',
    });
  }, [edges, nodes, setWorkflowNotice]);

  const handleLoadWorkflowClick = useCallback(() => {
    setShowWorkflowMenu(false);
    workflowFileInputRef.current?.click();
  }, []);

  const handleWorkflowFileSelected = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      try {
        const snapshot = parseWorkflowSnapshot(await file.text());
        replaceWorkflow(snapshot.nodes, snapshot.edges);
        setSavedWorkflowFingerprint(getEditableWorkflowFingerprint(snapshot.nodes, snapshot.edges));
        setShowWorkflowMenu(false);
        setShowTemplatePanel(false);
        setShowValidationPanel(false);
        setWorkflowNotice({
          type: 'success',
          message: `Loaded workflow with ${snapshot.nodes.length} nodes.`,
        });
      } catch (err) {
        setWorkflowNotice({
          type: 'error',
          message: err instanceof Error ? err.message : 'Could not load workflow file.',
        });
      } finally {
        event.target.value = '';
      }
    },
    [replaceWorkflow, setWorkflowNotice],
  );

  const handleClear = useCallback(() => {
    if (nodes.length > 0 && !confirm('Clear all nodes and edges?')) return;
    clearWorkflow();
  }, [nodes, clearWorkflow]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeIds([node.id]);
      setSelectedNodeId(node.id);
      setNodeContextMenu(null);
      setShowWorkflowMenu(false);
    },
    [setSelectedNodeId],
  );

  const handleEdgeClick = useCallback(
    (_: React.MouseEvent, edge: { id: string }) => {
      setSelectedNodeIds([]);
      setSelectedEdgeId(edge.id);
      setNodeContextMenu(null);
      setSuggestionMenu(null);
      setShowWorkflowMenu(false);
    },
    [setSelectedEdgeId],
  );

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node<BaseNodeData>) => {
      event.preventDefault();
      event.stopPropagation();

      const contract = getNodeContract(node.type);
      setSelectedNodeIds([node.id]);
      setSelectedNodeId(node.id);
      setSelectedEdgeId(null);
      setSuggestionMenu(null);
      setShowTemplatePanel(false);
      setShowWorkflowMenu(false);
      setNodeContextMenu({
        nodeId: node.id,
        nodeType: node.type,
        label: String(node.data.label || contract?.label || node.type || 'Node'),
        icon: contract?.icon || '*',
        category: contract?.category,
        hasDownstream: edges.some((edge) => edge.source === node.id),
        position: getNodeContextMenuPosition(event.clientX, event.clientY),
      });
    },
    [edges, setSelectedEdgeId, setSelectedNodeId],
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setSelectedNodeIds([]);
    setNodeContextMenu(null);
    setSuggestionMenu(null);
    setShowWorkflowMenu(false);
  }, [setSelectedEdgeId, setSelectedNodeId]);

  const handlePaneContextMenu = useCallback((event: MouseEvent | React.MouseEvent<Element, MouseEvent>) => {
    event.preventDefault();
    setNodeContextMenu(null);
    setSuggestionMenu(null);
  }, []);

  const handleCanvasMoveStart = useCallback(() => {
    setNodeContextMenu(null);
  }, []);

  const handleCanvasMove = useCallback((_event: MouseEvent | TouchEvent | null, viewport: Viewport) => {
    setViewportZoom(viewport.zoom);
    setTutorialViewportSignal((value) => value + 1);
  }, []);

  const handleNodeDragStart = useCallback(() => {
    setNodeContextMenu(null);
    checkpointHistory();
    setTutorialViewportSignal((value) => value + 1);
  }, [checkpointHistory]);

  const handleUndoClick = useCallback(() => {
    undoWorkflow();
    setSelectedNodeIds([]);
    setNodeContextMenu(null);
    setSuggestionMenu(null);
    setShowWorkflowMenu(false);
  }, [undoWorkflow]);

  const handleRedoClick = useCallback(() => {
    redoWorkflow();
    setSelectedNodeIds([]);
    setNodeContextMenu(null);
    setSuggestionMenu(null);
    setShowWorkflowMenu(false);
  }, [redoWorkflow]);

  const handleValidationIssueClick = useCallback(
    (issue: WorkflowIssue) => {
      if (issue.nodeId) {
        setSelectedNodeIds([issue.nodeId]);
        setSelectedNodeId(issue.nodeId);
      }
    },
    [setSelectedNodeId],
  );

  const handleToggleTemplates = useCallback(() => {
    setShowTemplatePanel((current) => !current);
    setShowValidationPanel(false);
    setNodeContextMenu(null);
    setShowWorkflowMenu(false);
  }, []);

  const handleToggleWorkflowMenu = useCallback(() => {
    setShowWorkflowMenu((current) => !current);
    setNodeContextMenu(null);
    setShowTemplatePanel(false);
    setShowValidationPanel(false);
  }, []);

  const handleSelectionChange = useCallback(
    ({
      nodes: selectedNodes,
      edges: selectedEdges,
    }: {
      nodes: Array<{ id: string }>;
      edges: Array<{ id: string }>;
    }) => {
      const nodeIds = selectedNodes.map((node) => node.id);
      setSelectedNodeIds(nodeIds);
      setSelectedNodeId(nodeIds[0] ?? null);
      setSelectedEdgeId(selectedNodes.length === 0 ? selectedEdges[0]?.id ?? null : null);
    },
    [setSelectedEdgeId, setSelectedNodeId],
  );

  const getFragmentForNodeIds = useCallback((nodeIds: string[]): ClipboardFragment | null => {
    const selectedIdSet = new Set(nodeIds);
    const fragmentNodes = nodes.filter((node) => selectedIdSet.has(node.id));

    if (fragmentNodes.length === 0) return null;

    const fragmentEdges = edges.filter((edge) =>
      selectedIdSet.has(edge.source) && selectedIdSet.has(edge.target),
    );
    const snapshot = createWorkflowSnapshot(fragmentNodes, fragmentEdges);

    return {
      nodes: snapshot.nodes,
      edges: snapshot.edges,
    };
  }, [edges, nodes]);

  const getSelectedFragment = useCallback((): ClipboardFragment | null => {
    const activeNodeIds = selectedNodeIds.length > 0
      ? selectedNodeIds
      : selectedNodeId
        ? [selectedNodeId]
        : [];
    const fragment = getFragmentForNodeIds(activeNodeIds);

    if (!fragment && selectedNodeId) {
      return getFragmentForNodeIds([selectedNodeId]);
    }

    return fragment;
  }, [getFragmentForNodeIds, selectedNodeId, selectedNodeIds]);

  const pasteFragment = useCallback(
    (fragment: ClipboardFragment, offset: { x: number; y: number }, message: string) => {
      const idByOriginal = new Map<string, string>();
      const pastedNodes = fragment.nodes.map((node) => {
        const nextId = generateNodeId();
        idByOriginal.set(node.id, nextId);

        return {
          ...node,
          id: nextId,
          position: {
            x: node.position.x + offset.x,
            y: node.position.y + offset.y,
          },
          data: cloneJson(node.data),
          selected: true,
          dragging: false,
        };
      });

      const pastedEdges = fragment.edges.flatMap((edge, index) => {
        const source = idByOriginal.get(edge.source);
        const target = idByOriginal.get(edge.target);
        if (!source || !target) return [];

        return [{
          ...edge,
          id: `edge_${source}_${target}_${index + 1}`,
          source,
          target,
          sourceHandle: edge.sourceHandle ?? null,
          targetHandle: edge.targetHandle ?? null,
          type: 'typed',
          animated: true,
          selected: false,
          data: cloneJson(edge.data),
        }];
      });

      insertGraphFragment(pastedNodes, pastedEdges);
      setSelectedNodeIds(pastedNodes.map((node) => node.id));
      setSuggestionMenu(null);
      setShowTemplatePanel(false);
      setShowWorkflowMenu(false);
      setWorkflowNotice({
        type: 'success',
        message,
      });
    },
    [insertGraphFragment, setWorkflowNotice],
  );

  const handleDuplicateSelectedNodes = useCallback(() => {
    const fragment = getSelectedFragment();
    if (!fragment) {
      setWorkflowNotice({
        type: 'info',
        message: 'Select one or more nodes before duplicating.',
      });
      return;
    }

    pasteFragment(
      fragment,
      { x: 36, y: 36 },
      `Duplicated ${fragment.nodes.length} node${fragment.nodes.length === 1 ? '' : 's'}.`,
    );
  }, [getSelectedFragment, pasteFragment, setWorkflowNotice]);

  const handleCopySingleNode = useCallback((nodeId: string) => {
    const fragment = getFragmentForNodeIds([nodeId]);
    if (!fragment) return;

    clipboardRef.current = cloneJson(fragment);
    setNodeContextMenu(null);
    setWorkflowNotice({
      type: 'success',
      message: 'Copied 1 node.',
    });
  }, [getFragmentForNodeIds, setWorkflowNotice]);

  const handleDuplicateSingleNode = useCallback((nodeId: string) => {
    const fragment = getFragmentForNodeIds([nodeId]);
    if (!fragment) return;

    pasteFragment(
      fragment,
      { x: 36, y: 36 },
      'Duplicated 1 node.',
    );
    setNodeContextMenu(null);
  }, [getFragmentForNodeIds, pasteFragment]);

  const handleDeleteSingleNode = useCallback((nodeId: string) => {
    removeNode(nodeId);
    setSelectedNodeIds([]);
    setNodeContextMenu(null);
  }, [removeNode]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isEditableShortcutTarget(event.target)) return;

      if (event.key === 'Escape') {
        setActiveTaskTutorial(null);
        setNodeContextMenu(null);
        setSuggestionMenu(null);
        setShowWorkflowMenu(false);
        return;
      }

      const isModifierPressed = event.ctrlKey || event.metaKey;
      if (!isModifierPressed) return;

      const key = event.key.toLowerCase();
      if (key === 'z') {
        event.preventDefault();
        if (event.shiftKey) {
          handleRedoClick();
        } else {
          handleUndoClick();
        }
        return;
      }

      if (key === 'y') {
        event.preventDefault();
        handleRedoClick();
        return;
      }

      if (key === 'd') {
        event.preventDefault();
        handleDuplicateSelectedNodes();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    handleDuplicateSelectedNodes,
    handleRedoClick,
    handleUndoClick,
  ]);

  const compatibleSuggestions = useMemo(() => {
    if (!suggestionMenu) return [];
    const allowed = suggestionMenu.direction === 'output'
      ? getAllowedTargets(suggestionMenu.sourceNodeType)
      : getAllowedSources(suggestionMenu.sourceNodeType);
    return nodePaletteItems.filter((item) => allowed.includes(item.type));
  }, [suggestionMenu]);

  const compatibleExistingNodes = useMemo(() => {
    if (!suggestionMenu) return [];
    const allowed = suggestionMenu.direction === 'output'
      ? getAllowedTargets(suggestionMenu.sourceNodeType)
      : getAllowedSources(suggestionMenu.sourceNodeType);

    return nodes.filter((node) => {
      if (node.id === suggestionMenu.sourceNodeId) return false;
      if (!node.type || !allowed.includes(node.type)) return false;

      if (suggestionMenu.direction === 'output') {
        return !edges.some(
          (edge) =>
            edge.source === suggestionMenu.sourceNodeId &&
            edge.target === node.id,
        );
      }

      return !edges.some(
        (edge) =>
          edge.source === node.id &&
          edge.target === suggestionMenu.sourceNodeId,
      );
    });
  }, [edges, nodes, suggestionMenu]);

  const suggestionMenuPosition = useMemo(() => {
    if (!suggestionMenu) return null;

    const sourceNode = nodes.find((node) => node.id === suggestionMenu.sourceNodeId);
    if (!sourceNode) return null;

    const sourceWidth = sourceNode.width ?? 320;
    const x = suggestionMenu.direction === 'output'
      ? sourceNode.position.x + sourceWidth + SUGGESTION_MENU_GAP
      : sourceNode.position.x - SUGGESTION_MENU_WIDTH - SUGGESTION_MENU_GAP;

    return {
      x,
      y: sourceNode.position.y - 12,
    };
  }, [nodes, suggestionMenu]);

  const handleConnectExistingNode = useCallback(
    (nodeId: string) => {
      if (!suggestionMenu) return;

      if (suggestionMenu.direction === 'output') {
        onConnect({
          source: suggestionMenu.sourceNodeId,
          target: nodeId,
          sourceHandle: null,
          targetHandle: null,
        });
      } else {
        onConnect({
          source: nodeId,
          target: suggestionMenu.sourceNodeId,
          sourceHandle: null,
          targetHandle: null,
        });
      }

      setSelectedNodeId(nodeId);
      setSuggestionMenu(null);
    },
    [onConnect, setSelectedNodeId, suggestionMenu],
  );

  const handleCreateSuggestedNode = useCallback(
    (nodeType: string, defaultData: Record<string, unknown>) => {
      if (!suggestionMenu) return;

      const sourceNode = nodes.find((node) => node.id === suggestionMenu.sourceNodeId);
      if (!sourceNode) return;

      const newNodeId = generateNodeId();
      const defaultDataCopy = JSON.parse(JSON.stringify(defaultData));
      const newNode = {
        id: newNodeId,
        type: nodeType,
        position: {
          x: sourceNode.position.x + (suggestionMenu.direction === 'output' ? 390 : -390),
          y: sourceNode.position.y,
        },
        style: nodeType === 'interactiveAnnotator' ? { width: 430, height: 560 } : undefined,
        data: defaultDataCopy,
      };

      addNode(newNode);
      if (suggestionMenu.direction === 'output') {
        onConnect({
          source: suggestionMenu.sourceNodeId,
          target: newNodeId,
          sourceHandle: null,
          targetHandle: null,
        });
      } else {
        onConnect({
          source: newNodeId,
          target: suggestionMenu.sourceNodeId,
          sourceHandle: null,
          targetHandle: null,
        });
      }
      setSelectedNodeId(newNodeId);
      setSuggestionMenu(null);
    },
    [addNode, nodes, onConnect, setSelectedNodeId, suggestionMenu],
  );

  const cloneDefaultData = useCallback((nodeType: string) => {
    const item = nodePaletteItems.find((paletteNode) => paletteNode.type === nodeType);
    return item ? JSON.parse(JSON.stringify(item.defaultData)) : { label: nodeType, status: 'idle' };
  }, []);

  const startCrowdsourcingTask = useCallback(
    (
      task: api.CrowdsourcingTask,
      userId: string,
      remainingTasks: api.CrowdsourcingTask[] = crowdsourcingTaskQueue,
      assignedTasks: api.CrowdsourcingTask[] = crowdsourcingSession?.assignedTasks || [],
    ) => {
      const loadPath = getCrowdsourcingTaskLoadPath(task);
      if (!loadPath) {
        setWorkflowNotice({
          type: 'error',
          message: 'The selected assignment does not include a loadable patient path.',
        });
        return;
      }

      const loaderId = generateNodeId();
      const annotatorId = generateNodeId();
      const tasksId = generateNodeId();
      const nextNodes: Node<BaseNodeData>[] = [
        {
          id: loaderId,
          type: 'dataLoader',
          position: { x: 80, y: 120 },
          data: {
            ...cloneDefaultData('dataLoader'),
            path: loadPath,
          },
        },
        {
          id: annotatorId,
          type: 'interactiveAnnotator',
          position: { x: 430, y: 90 },
          style: { width: 430, height: 560 },
          data: {
            ...cloneDefaultData('interactiveAnnotator'),
            sourcePath: loadPath,
            userId,
          },
        },
        {
          id: tasksId,
          type: 'crowdsourcingTasks',
          position: { x: 900, y: 190 },
          data: {
            ...cloneDefaultData('crowdsourcingTasks'),
            userId,
            currentPatientId: task.patient_id,
            tasks: buildCrowdsourcingTaskItems(assignedTasks, remainingTasks, task),
          },
        },
      ];
      const nextEdges: Edge[] = [
        {
          id: `edge_${loaderId}_${annotatorId}`,
          source: loaderId,
          target: annotatorId,
          sourceHandle: null,
          targetHandle: null,
          type: 'typed',
          animated: true,
          data: getConnectionEdgeData('dataLoader', 'interactiveAnnotator'),
        },
      ];

      replaceWorkflow(nextNodes, nextEdges);
      setSavedWorkflowFingerprint(getEditableWorkflowFingerprint(nextNodes, nextEdges));
      setCurrentCrowdsourcingTask(task);
      setSelectedNodeIds([annotatorId]);
      setSelectedNodeId(annotatorId);
      setShowTemplatePanel(false);
      setShowValidationPanel(false);
      setShowWorkflowMenu(false);
      setCrowdsourcingPanelMinimized(false);
      setCrowdsourcingAutoRunToken(Date.now());
      setWorkflowNotice({
        type: 'success',
        message: `Loaded assigned task ${task.patient_id} for ${userId}.`,
      });
    },
    [cloneDefaultData, crowdsourcingSession, crowdsourcingTaskQueue, replaceWorkflow, setSelectedNodeId, setWorkflowNotice],
  );

  const loadAdminWorkspace = useCallback(async (mode: 'manage' | 'review') => {
    try {
      const res = await api.getCollaborationCampaigns();
      const firstCampaign = res.campaigns[0];
      const workflowCampaign = firstCampaign ? mapWorkflowCampaign(firstCampaign) : null;

      if (mode === 'review') {
        const statusId = generateNodeId();
        const nextNodes: Node<BaseNodeData>[] = [
          {
            id: statusId,
            type: 'campaignStatus',
            position: { x: 820, y: 190 },
            data: {
              ...cloneDefaultData('campaignStatus'),
              campaignName: firstCampaign?.name || '',
              campaign: workflowCampaign,
            },
          },
        ];

        replaceWorkflow(nextNodes, []);
        setSavedWorkflowFingerprint(getEditableWorkflowFingerprint(nextNodes, []));
        setSelectedNodeIds([statusId]);
        setSelectedNodeId(statusId);
        setWorkflowNotice({
          type: 'success',
          message: firstCampaign
            ? `Loaded review status for ${firstCampaign.name}.`
            : 'No campaigns found. Create a campaign before reviewing submissions.',
        });
        return;
      }

      const setupId = generateNodeId();
      const assignId = generateNodeId();
      const statusId = generateNodeId();
      const nextNodes: Node<BaseNodeData>[] = [
        {
          id: setupId,
          type: 'campaignSetup',
          position: { x: 80, y: 150 },
          data: {
            ...cloneDefaultData('campaignSetup'),
            campaignName: firstCampaign?.name || '',
            datasetPath: firstCampaign?.dataset_path || '',
            totalPatients: firstCampaign?.total_patients || 0,
            patients: firstCampaign?.patients || [],
            campaign: workflowCampaign,
          },
        },
        {
          id: assignId,
          type: 'patientAssign',
          position: { x: 430, y: 150 },
          data: {
            ...cloneDefaultData('patientAssign'),
            campaignName: firstCampaign?.name || '',
            assignmentMode: 'allUnassigned',
            campaign: workflowCampaign,
            unassignedPatients: firstCampaign?.unassigned_patients || [],
          },
        },
        {
          id: statusId,
          type: 'campaignStatus',
          position: { x: 780, y: 150 },
          data: {
            ...cloneDefaultData('campaignStatus'),
            campaignName: firstCampaign?.name || '',
            campaign: workflowCampaign,
          },
        },
      ];
      const nextEdges: Edge[] = [
        {
          id: `edge_${setupId}_${assignId}`,
          source: setupId,
          target: assignId,
          type: 'typed',
          animated: true,
          data: getConnectionEdgeData('campaignSetup', 'patientAssign'),
        },
        {
          id: `edge_${assignId}_${statusId}`,
          source: assignId,
          target: statusId,
          type: 'typed',
          animated: true,
          data: getConnectionEdgeData('patientAssign', 'campaignStatus'),
        },
      ];

      replaceWorkflow(nextNodes, nextEdges);
      setSavedWorkflowFingerprint(getEditableWorkflowFingerprint(nextNodes, nextEdges));
      setSelectedNodeIds([setupId]);
      setSelectedNodeId(setupId);
      setWorkflowNotice({
        type: 'success',
        message: firstCampaign
          ? `Loaded campaign management for ${firstCampaign.name}.`
          : 'Loaded campaign management. Expand Campaign Setup to create the first campaign.',
      });
    } catch (error) {
      setWorkflowNotice({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to load admin crowdsourcing workspace.',
      });
    }
  }, [cloneDefaultData, replaceWorkflow, setSelectedNodeId, setWorkflowNotice]);

  const loadCrowdsourcingWorkspace = useCallback(
    (session: CrowdsourcingSession) => {
      setShowWorkflowMenu(false);
      setShowTemplatePanel(false);
      setNodeContextMenu(null);

      if (session.role === 'admin') {
        void loadAdminWorkspace('manage');
        setWorkflowNotice({
          type: 'success',
          message: 'Admin workspace loaded. Use Collaboration nodes to manage campaigns and assignments.',
        });
        return;
      }

      const guideDismissed = typeof window !== 'undefined' &&
        window.localStorage.getItem(CROWDSOURCING_GUIDE_DISMISSED_KEY) === 'true';
      if (!guideDismissed) {
        setDoNotShowCrowdsourcingGuide(false);
        setShowCrowdsourcingGuide(true);
      }

      const queue = crowdsourcingTaskQueue.length > 0
        ? crowdsourcingTaskQueue
        : session.remainingTasks;
      const task = currentCrowdsourcingTask || queue[0];

      if (task) {
        startCrowdsourcingTask(task, session.userId, queue, session.assignedTasks);
      } else {
        setCurrentCrowdsourcingTask(null);
        setWorkflowNotice({
          type: 'info',
          message: 'No remaining crowdsourcing tasks are assigned to this expert.',
        });
      }
    },
    [
      crowdsourcingTaskQueue,
      currentCrowdsourcingTask,
      loadAdminWorkspace,
      setWorkflowNotice,
      startCrowdsourcingTask,
    ],
  );

  const handleCrowdsourcingLoginSuccess = useCallback(
    (login: api.LoginResponse) => {
      const session: CrowdsourcingSession = {
        userId: login.user_id,
        role: login.role,
        expertScore: login.expert_score,
        remainingTasks: login.remaining_tasks,
        assignedTasks: login.assigned_tasks,
      };
      setCrowdsourcingSession(session);
      setCrowdsourcingTaskQueue(login.remaining_tasks);
      setShowCrowdsourcingLogin(false);
      setCrowdsourcingPanelMinimized(false);

      if (crowdsourcingLoginIntent === 'crowdsourcing') {
        setShowCrowdsourcingConfirm(true);
        return;
      }

      setWorkflowNotice({
        type: 'success',
        message: `${login.role === 'admin' ? 'Admin' : 'User'} login successful.`,
      });
    },
    [crowdsourcingLoginIntent, setWorkflowNotice],
  );

  const handleOpenLogin = useCallback(() => {
    setCrowdsourcingLoginIntent('session');
    setShowCrowdsourcingLogin(true);
    setShowCrowdsourcingConfirm(false);
    setShowWorkflowMenu(false);
    setShowTemplatePanel(false);
    setNodeContextMenu(null);
  }, []);

  const handleOpenCrowdsourcing = useCallback(() => {
    setShowWorkflowMenu(false);
    setShowTemplatePanel(false);
    setNodeContextMenu(null);

    if (!crowdsourcingSession) {
      setCrowdsourcingLoginIntent('crowdsourcing');
      setShowCrowdsourcingLogin(true);
      return;
    }

    setShowCrowdsourcingConfirm(true);
  }, [crowdsourcingSession]);

  const handleConfirmCrowdsourcingNavigation = useCallback(() => {
    if (!crowdsourcingSession) return;
    setShowCrowdsourcingConfirm(false);
    loadCrowdsourcingWorkspace(crowdsourcingSession);
  }, [crowdsourcingSession, loadCrowdsourcingWorkspace]);

  const handleCrowdsourcingLogout = useCallback(() => {
    setCrowdsourcingSession(null);
    setCurrentCrowdsourcingTask(null);
    setCrowdsourcingTaskQueue([]);
    setCrowdsourcingPanelMinimized(false);
    setShowCrowdsourcingLogin(false);
    setShowCrowdsourcingConfirm(false);
    setCrowdsourcingTaskConfirm(null);
    clearWorkflow();
    setSelectedNodeIds([]);
    setSelectedNodeId(null);
    setSavedWorkflowFingerprint(getEditableWorkflowFingerprint([], []));
    setWorkflowNotice({
      type: 'info',
      message: 'Crowdsourcing session closed.',
    });
  }, [clearWorkflow, setSelectedNodeId, setWorkflowNotice]);

  const handleRequestSubmitCrowdsourcingTask = useCallback(() => {
    if (!crowdsourcingSession || !currentCrowdsourcingTask) return;
    setCrowdsourcingTaskConfirm('submit');
  }, [crowdsourcingSession, currentCrowdsourcingTask]);

  const handleRequestNextCrowdsourcingTask = useCallback(() => {
    if (!crowdsourcingSession || !currentCrowdsourcingTask) return;
    setCrowdsourcingTaskConfirm('next');
  }, [crowdsourcingSession, currentCrowdsourcingTask]);

  const handleStartNextCrowdsourcingTask = useCallback(() => {
    if (!crowdsourcingSession) return;
    const currentIndex = crowdsourcingTaskQueue.findIndex((task) =>
      task.campaign_id === currentCrowdsourcingTask?.campaign_id &&
      task.patient_id === currentCrowdsourcingTask?.patient_id,
    );
    const nextIndex = currentIndex >= 0
      ? (currentIndex + 1) % crowdsourcingTaskQueue.length
      : 0;
    const nextTask = crowdsourcingTaskQueue[nextIndex];

    if (!nextTask) {
      setWorkflowNotice({
        type: 'info',
        message: 'No remaining crowdsourcing tasks are available.',
      });
      return;
    }

    startCrowdsourcingTask(nextTask, crowdsourcingSession.userId, crowdsourcingTaskQueue, crowdsourcingSession.assignedTasks);
  }, [
    crowdsourcingSession,
    crowdsourcingTaskQueue,
    currentCrowdsourcingTask,
    setWorkflowNotice,
    startCrowdsourcingTask,
  ]);

  const handleCloseCrowdsourcingGuide = useCallback(() => {
    if (doNotShowCrowdsourcingGuide && typeof window !== 'undefined') {
      window.localStorage.setItem(CROWDSOURCING_GUIDE_DISMISSED_KEY, 'true');
    }
    setShowCrowdsourcingGuide(false);
  }, [doNotShowCrowdsourcingGuide]);

  const handleSetCrowdsourcingHelperHidden = useCallback((hidden: boolean) => {
    setCrowdsourcingHelperHidden(hidden);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(CROWDSOURCING_HELPER_HIDDEN_KEY, hidden ? 'true' : 'false');
    }
  }, []);

  const handleSetDeidentificationHelperHidden = useCallback((hidden: boolean) => {
    setDeidentificationHelperHidden(hidden);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(DEIDENTIFICATION_HELPER_HIDDEN_KEY, hidden ? 'true' : 'false');
    }
  }, []);

  const handleCloseDeidentificationGuide = useCallback(() => {
    if (doNotShowDeidentificationGuide && typeof window !== 'undefined') {
      window.localStorage.setItem(DEIDENTIFICATION_GUIDE_DISMISSED_KEY, 'true');
    }
    setShowDeidentificationGuide(false);
  }, [doNotShowDeidentificationGuide]);

  const handleCloseDeidentificationResult = useCallback(() => {
    if (doNotShowDeidentificationResult && typeof window !== 'undefined') {
      window.localStorage.setItem(DEIDENTIFICATION_RESULT_DISMISSED_KEY, 'true');
    }
    setShowDeidentificationResult(false);
  }, [doNotShowDeidentificationResult]);

  const handleCloseTaskTutorial = useCallback(() => {
    if (activeTaskTutorial && doNotShowTaskTutorial && typeof window !== 'undefined') {
      window.localStorage.setItem(activeTaskTutorial.storageKey, 'true');
    }
    setActiveTaskTutorial(null);
  }, [activeTaskTutorial, doNotShowTaskTutorial]);

  const handleSubmitCrowdsourcingTask = useCallback(async () => {
    if (!crowdsourcingSession || !currentCrowdsourcingTask) return;
    if (crowdsourcingSubmitBusy) return;

    setCrowdsourcingSubmitBusy(true);
    try {
      const annotationData = buildCrowdsourcingAnnotationData(
        nodes,
        currentCrowdsourcingTask,
        crowdsourcingSession.userId,
      );
      await api.completeCollaborationTask({
        campaign_name: currentCrowdsourcingTask.campaign_id,
        expert_id: crowdsourcingSession.userId,
        patient_id: currentCrowdsourcingTask.patient_id,
        annotation_data: annotationData,
      });

      const refreshed = await api.getCollaborationExpertTasks(crowdsourcingSession.userId, true);
      const remainingTasks = refreshed.tasks.filter((task) =>
        task.campaign_id !== currentCrowdsourcingTask.campaign_id ||
        task.patient_id !== currentCrowdsourcingTask.patient_id,
      );

      setCrowdsourcingTaskQueue(remainingTasks);
      setCrowdsourcingSession({
        ...crowdsourcingSession,
        remainingTasks,
      });

      const nextTask = remainingTasks[0];
      if (nextTask) {
        setWorkflowNotice({
          type: 'success',
          message: `Submitted ${currentCrowdsourcingTask.patient_id}. Loading next assignment.`,
        });
        startCrowdsourcingTask(nextTask, crowdsourcingSession.userId, remainingTasks, crowdsourcingSession.assignedTasks);
      } else {
        setCurrentCrowdsourcingTask(null);
        setWorkflowNotice({
          type: 'success',
          message: `Submitted ${currentCrowdsourcingTask.patient_id}. No remaining assignments.`,
        });
      }
    } catch (error) {
      setWorkflowNotice({
        type: 'error',
        message: error instanceof Error ? error.message : 'Could not submit crowdsourcing task.',
      });
    } finally {
      setCrowdsourcingSubmitBusy(false);
    }
  }, [
    crowdsourcingSession,
    crowdsourcingSubmitBusy,
    currentCrowdsourcingTask,
    nodes,
    setWorkflowNotice,
    startCrowdsourcingTask,
  ]);

  const handleConfirmCrowdsourcingTaskAction = useCallback(() => {
    if (crowdsourcingTaskConfirm === 'submit') {
      setCrowdsourcingTaskConfirm(null);
      void handleSubmitCrowdsourcingTask();
      return;
    }

    if (crowdsourcingTaskConfirm === 'next') {
      setCrowdsourcingTaskConfirm(null);
      handleStartNextCrowdsourcingTask();
    }
  }, [crowdsourcingTaskConfirm, handleStartNextCrowdsourcingTask, handleSubmitCrowdsourcingTask]);

  useEffect(() => {
    if (!crowdsourcingAutoRunToken || isWorkflowRunning || nodes.length === 0) return;

    const timeout = window.setTimeout(() => {
      setCrowdsourcingAutoRunToken(0);
      void handleRun('all');
    }, 0);

    return () => window.clearTimeout(timeout);
  }, [crowdsourcingAutoRunToken, handleRun, isWorkflowRunning, nodes.length]);

  const handleCreateDataLoader = useCallback(() => {
    addNode({
      id: generateNodeId(),
      type: 'dataLoader',
      position: { x: 80, y: 90 },
      data: cloneDefaultData('dataLoader'),
    });
  }, [addNode, cloneDefaultData]);

  const insertTemplate = useCallback(
    (template: WorkflowTemplate) => {
      clearWorkflow();
      const instance = instantiateWorkflowTemplate(template, generateNodeId);
      const insertedNodeIds = instance.nodes.map((node) => node.id);

      addNodesAndConnect(instance.nodes, instance.connections);
      setShowTemplatePanel(false);
      setShowValidationPanel(false);
      const tutorial = taskTutorials[template.id];
      const dismissed = tutorial && typeof window !== 'undefined' &&
        window.localStorage.getItem(tutorial.storageKey) === 'true';
      const shouldShowTutorial = Boolean(tutorial && !dismissed);
      const shouldFitInsertedTask = shouldShowTutorial ||
        template.id === MEDICAL_REPORT_TEMPLATE_ID;

      if (shouldFitInsertedTask) {
        window.setTimeout(() => {
          reactFlowInstance.current?.fitView?.({
            nodes: insertedNodeIds.map((id) => ({ id })),
            padding: 0.22,
            duration: 350,
          });
        }, 80);
      }

      if (tutorial && !dismissed) {
        setDoNotShowTaskTutorial(false);
        window.setTimeout(() => {
          setTutorialViewportSignal((value) => value + 1);
          setActiveTaskTutorial(tutorial);
        }, 520);
      }
      setWorkflowNotice({
        type: 'success',
        message: `${template.title} template added.`,
      });
    },
    [addNodesAndConnect, clearWorkflow, setWorkflowNotice],
  );

  const handleApplyTemplate = useCallback(
    (template: WorkflowTemplate) => {
      setShowTaskStartModal(false);
      if (nodes.length > 0) {
        setPendingTemplate(template);
        setShowTemplatePanel(false);
        setShowValidationPanel(false);
        return;
      }

      insertTemplate(template);
    },
    [insertTemplate, nodes.length],
  );

  const handleProceedWithPendingTemplate = useCallback((saveFirst: boolean) => {
    if (!pendingTemplate) return;
    if (saveFirst) {
      handleSaveWorkflow();
    }
    const template = pendingTemplate;
    setPendingTemplate(null);
    insertTemplate(template);
  }, [handleSaveWorkflow, insertTemplate, pendingTemplate]);

  const getSuggestionEdgeData = useCallback(
    (candidateType: string | undefined) => {
      if (!suggestionMenu) return undefined;
      return suggestionMenu.direction === 'output'
        ? getConnectionEdgeData(suggestionMenu.sourceNodeType, candidateType)
        : getConnectionEdgeData(candidateType, suggestionMenu.sourceNodeType);
    },
    [suggestionMenu],
  );

  const nodeContextActions = useMemo(() => {
    if (!nodeContextMenu) return [];
    const runDisabled = isWorkflowRunning;
    const downstreamDisabled = isWorkflowRunning || !nodeContextMenu.hasDownstream;

    return [
      {
        id: 'run-selected',
        label: 'Run this node',
        description: runDisabled
          ? 'A workflow run is already in progress.'
          : `Only execute ${nodeContextMenu.label} using current settings and available upstream data.`,
        color: 'var(--accent-green)',
        disabled: runDisabled,
        onClick: () => { void handleRun('selected', nodeContextMenu.nodeId); },
      },
      {
        id: 'run-downstream',
        label: 'Run downstream branch',
        description: !nodeContextMenu.hasDownstream
          ? 'No downstream nodes are connected to this node.'
          : runDisabled
            ? 'A workflow run is already in progress.'
            : 'Execute this node, then every connected node after it.',
        color: 'var(--accent-purple)',
        disabled: downstreamDisabled,
        onClick: () => { void handleRun('downstream', nodeContextMenu.nodeId); },
      },
      {
        id: 'copy',
        label: 'Copy',
        description: 'Copy this node into the workflow clipboard.',
        color: 'var(--accent-blue)',
        onClick: () => handleCopySingleNode(nodeContextMenu.nodeId),
      },
      {
        id: 'duplicate',
        label: 'Duplicate',
        description: runDisabled
          ? 'Graph editing is disabled while a run is active.'
          : 'Create an offset copy of this node.',
        color: 'var(--accent-blue)',
        disabled: runDisabled,
        onClick: () => handleDuplicateSingleNode(nodeContextMenu.nodeId),
      },
      {
        id: 'delete',
        label: 'Delete',
        description: runDisabled
          ? 'Graph editing is disabled while a run is active.'
          : 'Remove this node and its connections. Undo is available.',
        color: 'var(--accent-red)',
        disabled: runDisabled,
        onClick: () => handleDeleteSingleNode(nodeContextMenu.nodeId),
      },
    ];
  }, [
    handleCopySingleNode,
    handleDeleteSingleNode,
    handleDuplicateSingleNode,
    handleRun,
    isWorkflowRunning,
    nodeContextMenu,
  ]);

  return (
    <div style={appStyle}>
      {showCrowdsourcingLogin ? (
        <CrowdsourcingLoginModal
          onClose={() => setShowCrowdsourcingLogin(false)}
          onSuccess={handleCrowdsourcingLoginSuccess}
        />
      ) : null}

      {/* Left sidebar — Node Palette */}
      {showCrowdsourcingConfirm && crowdsourcingSession ? (
        <div
          style={guideOverlayStyle}
          onClick={() => setShowCrowdsourcingConfirm(false)}
          role="presentation"
        >
          <div
            style={confirmModalStyle}
            role="dialog"
            aria-modal="true"
            aria-labelledby="crowdsourcing-confirm-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div style={{
              padding: '15px 17px 12px',
              borderBottom: '1px solid var(--border-color)',
            }}>
              <div
                id="crowdsourcing-confirm-title"
                style={{
                  color: crowdsourcingSession.role === 'admin'
                    ? 'var(--accent-blue)'
                    : 'var(--accent-green)',
                  fontSize: 14,
                  fontWeight: 900,
                  textTransform: 'uppercase',
                }}
              >
                Continue to Crowdsourcing
              </div>
            </div>
            <div style={{ padding: 17 }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.55 }}>
                {crowdsourcingSession.role === 'admin'
                  ? 'The crowdsourcing admin workspace is about to be loaded. Do you want to continue?'
                  : 'Your crowdsourcing tasks are about to be loaded. Do you want to continue?'}
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
                <button
                  type="button"
                  onClick={() => setShowCrowdsourcingConfirm(false)}
                  style={btnStyle('var(--text-secondary)')}
                  onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', true)}
                  onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmCrowdsourcingNavigation}
                  disabled={isWorkflowRunning}
                  style={btnStyle(
                    crowdsourcingSession.role === 'admin'
                      ? 'var(--accent-blue)'
                      : 'var(--accent-green)',
                    isWorkflowRunning,
                  )}
                  onMouseEnter={(event) => setButtonHover(
                    event.currentTarget,
                    crowdsourcingSession.role === 'admin'
                      ? 'var(--accent-blue)'
                      : 'var(--accent-green)',
                    true,
                    isWorkflowRunning,
                  )}
                  onMouseLeave={(event) => setButtonHover(
                    event.currentTarget,
                    crowdsourcingSession.role === 'admin'
                      ? 'var(--accent-blue)'
                      : 'var(--accent-green)',
                    false,
                    isWorkflowRunning,
                  )}
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {crowdsourcingTaskConfirm && crowdsourcingSession && currentCrowdsourcingTask ? (
        <div
          style={guideOverlayStyle}
          onClick={() => setCrowdsourcingTaskConfirm(null)}
          role="presentation"
        >
          <div
            style={confirmModalStyle}
            role="dialog"
            aria-modal="true"
            aria-labelledby="crowdsourcing-task-confirm-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div style={{
              padding: '15px 17px 12px',
              borderBottom: '1px solid var(--border-color)',
            }}>
              <div
                id="crowdsourcing-task-confirm-title"
                style={{
                  color: crowdsourcingTaskConfirm === 'submit'
                    ? 'var(--accent-green)'
                    : 'var(--accent-blue)',
                  fontSize: 14,
                  fontWeight: 900,
                  textTransform: 'uppercase',
                }}
              >
                {crowdsourcingTaskConfirm === 'submit' ? 'Submit Crowdsourcing Task' : 'Load Next Task'}
              </div>
            </div>
            <div style={{ padding: 17 }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.55 }}>
                {crowdsourcingTaskConfirm === 'submit'
                  ? 'This assignment will be marked completed and the current annotations will be saved to the crowdsourcing database.'
                  : 'The next assignment will be loaded without submitting this patient. Continue only if you want to skip submitting the current task for now.'}
              </div>

              <div style={{
                marginTop: 12,
                padding: '9px 10px',
                borderRadius: 7,
                border: '1px solid color-mix(in srgb, var(--accent-blue) 24%, var(--border-color))',
                background: 'color-mix(in srgb, var(--accent-blue) 7%, var(--bg-secondary))',
                display: 'grid',
                gap: 5,
                color: 'var(--text-secondary)',
                fontSize: 12,
              }}>
                <div><strong style={{ color: 'var(--text-primary)' }}>Patient:</strong> {currentCrowdsourcingTask.patient_id}</div>
                <div><strong style={{ color: 'var(--text-primary)' }}>Campaign:</strong> {currentCrowdsourcingTask.campaign_id}</div>
                <div><strong style={{ color: 'var(--text-primary)' }}>Expert:</strong> {crowdsourcingSession.userId}</div>
                {currentCrowdsourcingTask.modality ? (
                  <div><strong style={{ color: 'var(--text-primary)' }}>Modality:</strong> {currentCrowdsourcingTask.modality.toUpperCase()}</div>
                ) : null}
                <div><strong style={{ color: 'var(--text-primary)' }}>Annotations:</strong> {currentTaskAnnotationCount}</div>
                <div><strong style={{ color: 'var(--text-primary)' }}>Remaining queue:</strong> {crowdsourcingTaskQueue.length}</div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
                <button
                  type="button"
                  onClick={() => setCrowdsourcingTaskConfirm(null)}
                  style={btnStyle('var(--text-secondary)')}
                  onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', true)}
                  onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmCrowdsourcingTaskAction}
                  disabled={crowdsourcingSubmitBusy || isWorkflowRunning}
                  style={btnStyle(
                    crowdsourcingTaskConfirm === 'submit'
                      ? 'var(--accent-green)'
                      : 'var(--accent-blue)',
                    crowdsourcingSubmitBusy || isWorkflowRunning,
                  )}
                  onMouseEnter={(event) => setButtonHover(
                    event.currentTarget,
                    crowdsourcingTaskConfirm === 'submit'
                      ? 'var(--accent-green)'
                      : 'var(--accent-blue)',
                    true,
                    crowdsourcingSubmitBusy || isWorkflowRunning,
                  )}
                  onMouseLeave={(event) => setButtonHover(
                    event.currentTarget,
                    crowdsourcingTaskConfirm === 'submit'
                      ? 'var(--accent-green)'
                      : 'var(--accent-blue)',
                    false,
                    crowdsourcingSubmitBusy || isWorkflowRunning,
                  )}
                >
                  {crowdsourcingTaskConfirm === 'submit' ? 'Submit Task' : 'Load Next'}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {pendingTemplate && typeof document !== 'undefined'
        ? createPortal(
          <div style={guideOverlayStyle} onClick={() => setPendingTemplate(null)} role="presentation">
            <div
              style={confirmModalStyle}
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="replace-workflow-title"
            >
              <div style={{
                padding: '15px 16px',
                borderBottom: '1px solid var(--border-color)',
              }}>
                <div
                  id="replace-workflow-title"
                  style={{
                    color: 'var(--accent-orange)',
                    fontSize: 14,
                    fontWeight: 900,
                    textTransform: 'uppercase',
                  }}
                >
                  Grid Is Not Empty
                </div>
                <div style={{
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginTop: 7,
                }}>
                  Adding <strong style={{ color: 'var(--text-primary)' }}>{pendingTemplate.title}</strong> will clear the current grid before loading the task. Save the current workflow as JSON first, or proceed right away.
                </div>
              </div>
              <div style={{
                padding: 16,
                display: 'flex',
                justifyContent: 'flex-end',
                gap: 8,
                flexWrap: 'wrap',
              }}>
                <button
                  type="button"
                  onClick={() => setPendingTemplate(null)}
                  style={btnStyle('var(--text-secondary)')}
                  onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', true)}
                  onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--text-secondary)', false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => handleProceedWithPendingTemplate(false)}
                  style={btnStyle('var(--accent-orange)')}
                  onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-orange)', true)}
                  onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-orange)', false)}
                >
                  Proceed
                </button>
                <button
                  type="button"
                  onClick={() => handleProceedWithPendingTemplate(true)}
                  style={btnStyle('var(--accent-blue)')}
                  onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', true)}
                  onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', false)}
                >
                  Save JSON & Proceed
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )
        : null}

      {showTaskStartModal && typeof document !== 'undefined'
        ? createPortal(
          <div style={guideOverlayStyle} onClick={() => setShowTaskStartModal(false)} role="presentation">
            <div
              style={{
                width: 760,
                maxWidth: 'calc(100vw - 32px)',
                maxHeight: '82vh',
                borderRadius: 12,
                border: '1px solid color-mix(in srgb, var(--accent-green) 34%, var(--border-color))',
                background: 'var(--bg-secondary)',
                boxShadow: '0 22px 64px rgba(0, 0, 0, 0.52)',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
              }}
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="end-to-end-task-title"
            >
              <div style={{
                padding: '16px 18px',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: 16,
              }}>
                <div>
                  <div
                    id="end-to-end-task-title"
                    style={{
                      color: 'var(--accent-green)',
                      fontSize: 15,
                      fontWeight: 950,
                      textTransform: 'uppercase',
                    }}
                  >
                    End-to-end tasks
                  </div>
                  <div style={{
                    color: 'var(--text-secondary)',
                    fontSize: 12,
                    lineHeight: 1.5,
                    marginTop: 5,
                    maxWidth: 560,
                  }}>
                    Choose a ready workflow to start from. Each task loads a curated node graph and guided flow for a common medical imaging operation.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowTaskStartModal(false)}
                  style={{
                    ...iconButtonStyle('var(--text-secondary)'),
                    minHeight: 30,
                    width: 30,
                    minWidth: 30,
                  }}
                  title="Close"
                  aria-label="Close end-to-end tasks"
                >
                  x
                </button>
              </div>
              <div style={{
                padding: 16,
                overflowY: 'auto',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                gap: 10,
              }}>
                {workflowTemplates.map((template) => (
                  <div
                    key={template.id}
                    style={{
                      borderRadius: 8,
                      border: '1px solid color-mix(in srgb, var(--accent-green) 22%, var(--border-color))',
                      background: 'color-mix(in srgb, var(--accent-green) 6%, var(--bg-tertiary))',
                      padding: 12,
                      display: 'grid',
                      gap: 8,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'flex-start' }}>
                      <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 950 }}>
                        {template.title}
                      </div>
                      <div style={{
                        color: 'var(--accent-green)',
                        border: '1px solid rgba(76, 175, 139, 0.28)',
                        background: 'rgba(76, 175, 139, 0.09)',
                        borderRadius: 5,
                        padding: '2px 6px',
                        fontSize: 9,
                        fontWeight: 900,
                        whiteSpace: 'nowrap',
                      }}>
                        {template.nodes.length} nodes
                      </div>
                    </div>
                    {template.automationLevel ? (
                      <div style={{ color: 'var(--accent-green)', fontSize: 10, fontWeight: 850 }}>
                        {template.automationLevel}
                      </div>
                    ) : null}
                    <div style={{ color: 'var(--text-secondary)', fontSize: 11, lineHeight: 1.45 }}>
                      {template.description}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleApplyTemplate(template)}
                      style={{
                        ...compactButtonStyle('var(--accent-green)'),
                        justifySelf: 'start',
                      }}
                      onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', true)}
                      onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', false)}
                    >
                      Load task
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>,
          document.body,
        )
        : null}

      <NodePalette onApplyTemplate={handleApplyTemplate} />

      {/* Main canvas */}
      <div style={canvasContainerStyle}>
        {workflowNotice && (
          <div style={noticeStyle(workflowNotice.type)}>
            {workflowNotice.message}
          </div>
        )}

        {showValidationPanel && validationIssues.length > 0 ? (
          <WorkflowValidationPanel
            issues={validationIssues}
            onIssueClick={handleValidationIssueClick}
            onClose={() => setShowValidationPanel(false)}
          />
        ) : null}

        {showTemplatePanel ? (
          <WorkflowTemplatePanel
            templates={workflowTemplates}
            onApply={handleApplyTemplate}
            onClose={() => setShowTemplatePanel(false)}
          />
        ) : null}

        {nodeContextMenu ? (
          <NodeContextMenu
            title={nodeContextMenu.label}
            icon={nodeContextMenu.icon}
            category={nodeContextMenu.category}
            position={nodeContextMenu.position}
            actions={nodeContextActions}
            onClose={() => setNodeContextMenu(null)}
          />
        ) : null}

        {nodes.length === 0 ? (
          <CanvasEmptyState
            onCreateDataLoader={handleCreateDataLoader}
            onOpenEndToEndTasks={() => setShowTaskStartModal(true)}
          />
        ) : null}

        {hasDeidentificationWorkflow && !crowdsourcingSession ? (
          deidentificationHelperHidden ? (
            <button
              type="button"
              onClick={() => handleSetDeidentificationHelperHidden(false)}
              style={{
                ...btnStyle('var(--accent-green)'),
                position: 'absolute',
                top: 62,
                right: 12,
                zIndex: 9,
              }}
              title="Show deidentification guidance"
            >
              Show Tips
            </button>
          ) : (
            <div style={{
              ...crowdsourcingHelperStyle,
              border: '1px solid color-mix(in srgb, var(--accent-green) 35%, var(--border-color))',
              background: 'color-mix(in srgb, var(--accent-green) 8%, var(--bg-secondary))',
            }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ color: 'var(--accent-green)', fontSize: 11, fontWeight: 900, textTransform: 'uppercase' }}>
                  Deidentification Guidance
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.45, marginTop: 2 }}>
                  Load a study, run the workflow, then compare Metadata Before and Metadata After. The Deidentify node writes a new sanitized output and an audit entry without changing the source data.
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  setDoNotShowDeidentificationGuide(false);
                  setShowDeidentificationGuide(true);
                }}
                style={{ ...btnStyle('var(--accent-green)'), pointerEvents: 'auto' }}
              >
                Guide
              </button>
              <button
                type="button"
                onClick={() => handleSetDeidentificationHelperHidden(true)}
                style={{ ...btnStyle('var(--text-secondary)'), pointerEvents: 'auto' }}
              >
                Hide
              </button>
            </div>
          )
        ) : null}

        {crowdsourcingSession && currentCrowdsourcingTask ? (
          crowdsourcingHelperHidden ? (
            <button
              type="button"
              onClick={() => handleSetCrowdsourcingHelperHidden(false)}
              style={{
                ...btnStyle('var(--accent-blue)'),
                position: 'absolute',
                top: 62,
                right: 12,
                zIndex: 9,
              }}
              title="Show crowdsourcing guidance"
            >
              Show Tips
            </button>
          ) : (
            <div style={crowdsourcingHelperStyle}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ color: 'var(--accent-blue)', fontSize: 11, fontWeight: 900, textTransform: 'uppercase' }}>
                  Crowdsourcing Guidance
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.45, marginTop: 2 }}>
                  Work on the current patient in Interactive Annotator. Use the assignment table on the canvas to confirm the queue. Submit only when the patient annotation is ready; use Next Task to move without submitting.
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowCrowdsourcingGuide(true)}
                style={{ ...btnStyle('var(--accent-blue)'), pointerEvents: 'auto' }}
              >
                Guide
              </button>
              <button
                type="button"
                onClick={() => handleSetCrowdsourcingHelperHidden(true)}
                style={{ ...btnStyle('var(--text-secondary)'), pointerEvents: 'auto' }}
              >
                Hide
              </button>
            </div>
          )
        ) : null}

        {isAdminCrowdsourcingWorkspaceVisible ? (
          <div style={crowdsourcingHelperStyle}>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ color: 'var(--accent-blue)', fontSize: 11, fontWeight: 900, textTransform: 'uppercase' }}>
                Crowdsourcing Admin Guidance
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.45, marginTop: 2 }}>
                Manage campaigns, assign patients, and review submitted annotation progress from the workflow canvas.
              </div>
            </div>
            <button
              type="button"
              onClick={() => { void loadAdminWorkspace('manage'); }}
              style={{ ...compactButtonStyle('var(--accent-green)'), pointerEvents: 'auto' }}
            >
              Manage Campaigns
            </button>
            <button
              type="button"
              onClick={() => { void loadAdminWorkspace('review'); }}
              style={{ ...compactButtonStyle('var(--accent-blue)'), pointerEvents: 'auto' }}
            >
              Review Submissions
            </button>
          </div>
        ) : null}

        {crowdsourcingSession && currentCrowdsourcingTask ? (
          crowdsourcingPanelMinimized ? (
            <div
              style={{
                ...crowdsourcingMinimizedBaseStyle,
                transform: `scale(${crowdsourcingPanelScale})`,
              }}
            >
              <div style={{ color: 'var(--text-primary)', fontSize: 11, fontWeight: 900 }}>
                Task: {currentCrowdsourcingTask.patient_id}
              </div>
              <button
                type="button"
                onClick={() => setCrowdsourcingPanelMinimized(false)}
                style={compactButtonStyle('var(--accent-green)')}
              >
                Expand
              </button>
            </div>
          ) : (
            <div
              style={{
                ...crowdsourcingBannerBaseStyle,
                transform: `scale(${crowdsourcingPanelScale})`,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'flex-start' }}>
                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      color: 'var(--accent-green)',
                      fontSize: 11,
                      fontWeight: 900,
                      textTransform: 'uppercase',
                      letterSpacing: 0,
                    }}
                  >
                    Crowdsourcing Task
                  </div>
                  <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900, marginTop: 2 }}>
                    {currentCrowdsourcingTask.patient_id}
                  </div>
                  <div style={crowdsourcingMetaStyle}>
                    <span>{currentCrowdsourcingTask.campaign_id}</span>
                    <span>Expert: {crowdsourcingSession.userId}</span>
                    {currentCrowdsourcingTask.modality ? (
                      <span>{currentCrowdsourcingTask.modality.toUpperCase()}</span>
                    ) : null}
                    <span>{crowdsourcingTaskQueue.length} remaining</span>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => setCrowdsourcingPanelMinimized(true)}
                    style={compactIconButtonStyle('var(--text-secondary)')}
                    title="Minimize crowdsourcing task panel"
                    aria-label="Minimize crowdsourcing task panel"
                  >
                    -
                  </button>
                  <button
                    type="button"
                    onClick={handleCrowdsourcingLogout}
                    style={compactButtonStyle('var(--text-secondary)')}
                    title="Close crowdsourcing session"
                  >
                    Logout
                  </button>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button
                  type="button"
                  onClick={handleRequestSubmitCrowdsourcingTask}
                  disabled={crowdsourcingSubmitBusy || isWorkflowRunning}
                  style={compactButtonStyle('var(--accent-green)', crowdsourcingSubmitBusy || isWorkflowRunning)}
                  title="Mark this assignment as completed in db/assignments.json"
                >
                  {crowdsourcingSubmitBusy ? 'Submitting...' : 'Submit Task'}
                </button>
                <button
                  type="button"
                  onClick={handleRequestNextCrowdsourcingTask}
                  disabled={crowdsourcingTaskQueue.length === 0 || isWorkflowRunning}
                  style={compactButtonStyle('var(--accent-blue)', crowdsourcingTaskQueue.length === 0 || isWorkflowRunning)}
                >
                  Next Task
                </button>
              </div>
            </div>
          )
        ) : null}

        {/* Toolbar */}
        <div style={toolbarStyle}>
          <button
            type="button"
            onClick={
              crowdsourcingSession
                ? handleCrowdsourcingLogout
                : handleOpenLogin
            }
            disabled={isWorkflowRunning}
            style={btnStyle(
              crowdsourcingSession ? 'var(--text-secondary)' : 'var(--accent-blue)',
              isWorkflowRunning,
            )}
            onMouseEnter={(event) => setButtonHover(
              event.currentTarget,
              crowdsourcingSession ? 'var(--text-secondary)' : 'var(--accent-blue)',
              true,
              isWorkflowRunning,
            )}
            onMouseLeave={(event) => setButtonHover(
              event.currentTarget,
              crowdsourcingSession ? 'var(--text-secondary)' : 'var(--accent-blue)',
              false,
              isWorkflowRunning,
            )}
            title={
              crowdsourcingSession
                ? `Logout ${crowdsourcingSession.userId}`
                : 'Login with crowdsourcing credentials'
            }
          >
            {crowdsourcingSession ? 'Logout' : 'Login'}
          </button>
          {hasUnsavedChanges ? (
            <div style={unsavedBadgeStyle} title="Save Workflow to clear this indicator">
              Unsaved changes
            </div>
          ) : null}
          <button
            type="button"
            onClick={handleUndoClick}
            disabled={!canUndo}
            style={iconButtonStyle('var(--accent-blue)', !canUndo)}
            onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', true, !canUndo)}
            onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', false, !canUndo)}
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
          >
            {'\u21b6'}
          </button>
          <button
            type="button"
            onClick={handleRedoClick}
            disabled={!canRedo}
            style={iconButtonStyle('var(--accent-blue)', !canRedo)}
            onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', true, !canRedo)}
            onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', false, !canRedo)}
            title="Redo (Ctrl+Y)"
            aria-label="Redo"
          >
            {'\u21b7'}
          </button>
          <button
            type="button"
            onClick={handleOpenCrowdsourcing}
            disabled={isWorkflowRunning}
            style={btnStyle('var(--accent-purple)', isWorkflowRunning)}
            onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-purple)', true, isWorkflowRunning)}
            onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-purple)', false, isWorkflowRunning)}
            title={
              crowdsourcingSession
                ? `Open crowdsourcing for ${crowdsourcingSession.userId}`
                : 'Login and open crowdsourcing'
            }
          >
            Crowdsourcing
          </button>
          <button
            onClick={handleToggleTemplates}
            style={btnStyle('var(--accent-blue)', false, showTemplatePanel)}
            onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', true)}
            onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-blue)', false, false, showTemplatePanel)}
            title="Add a workflow template"
          >
            Templates
          </button>
          <div style={{ position: 'relative', display: 'flex' }}>
            <button
              type="button"
              onClick={() => { void handleRun('all'); }}
              disabled={nodes.length === 0 || isWorkflowRunning}
              style={splitRunButtonStyle(nodes.length === 0 || isWorkflowRunning)}
              onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', true, nodes.length === 0 || isWorkflowRunning)}
              onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', false, nodes.length === 0 || isWorkflowRunning)}
              title="Execute the workflow"
            >
              {isWorkflowRunning ? 'Running...' : 'Run Workflow'}
            </button>
            <button
              type="button"
              onClick={handleToggleWorkflowMenu}
              style={splitArrowButtonStyle(showWorkflowMenu)}
              onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', true)}
              onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-green)', false, false, showWorkflowMenu)}
              title="Save or load workflow"
              aria-label="Open workflow actions"
              aria-haspopup="menu"
              aria-expanded={showWorkflowMenu}
            >
              {showWorkflowMenu ? '^' : 'v'}
            </button>

          {showWorkflowMenu ? (
            <div style={workflowMenuStyle} role="menu">
              <button
                type="button"
                disabled={nodes.length === 0 || isWorkflowRunning}
                onClick={() => { void handleRun('all'); }}
                style={workflowMenuItemStyle('var(--accent-green)', nodes.length === 0 || isWorkflowRunning)}
                onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true, nodes.length === 0 || isWorkflowRunning)}
                onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false, nodes.length === 0 || isWorkflowRunning)}
                role="menuitem"
              >
                Run Full Workflow
              </button>
              <button
                type="button"
                disabled={!selectedExecutionNodeId || isWorkflowRunning}
                onClick={() => { void handleRun('selected'); }}
                style={workflowMenuItemStyle('var(--accent-blue)', !selectedExecutionNodeId || isWorkflowRunning)}
                onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true, !selectedExecutionNodeId || isWorkflowRunning)}
                onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false, !selectedExecutionNodeId || isWorkflowRunning)}
                role="menuitem"
              >
                Run Selected Node
              </button>
              <button
                type="button"
                disabled={!selectedExecutionNodeId || isWorkflowRunning}
                onClick={() => { void handleRun('downstream'); }}
                style={workflowMenuItemStyle('var(--accent-purple)', !selectedExecutionNodeId || isWorkflowRunning)}
                onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true, !selectedExecutionNodeId || isWorkflowRunning)}
                onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false, !selectedExecutionNodeId || isWorkflowRunning)}
                role="menuitem"
              >
                Run Downstream Branch
              </button>
              {isWorkflowRunning ? (
                <button
                  type="button"
                  onClick={handleCancelRun}
                  style={workflowMenuItemStyle('var(--accent-red)')}
                  onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true)}
                  onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false)}
                  role="menuitem"
                >
                  Cancel Current Run
                </button>
              ) : null}
              <button
                type="button"
                disabled={nodes.length === 0 || isWorkflowRunning}
                onClick={handleSaveWorkflow}
                style={workflowMenuItemStyle('var(--accent-blue)', nodes.length === 0 || isWorkflowRunning)}
                onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true, nodes.length === 0 || isWorkflowRunning)}
                onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false, nodes.length === 0 || isWorkflowRunning)}
                role="menuitem"
              >
                Save Workflow
              </button>
              <button
                type="button"
                disabled={isWorkflowRunning}
                onClick={handleLoadWorkflowClick}
                style={{
                  ...workflowMenuItemStyle('var(--accent-orange)', isWorkflowRunning),
                  borderBottom: 'none',
                }}
                onMouseEnter={(event) => setMenuItemHover(event.currentTarget, true, isWorkflowRunning)}
                onMouseLeave={(event) => setMenuItemHover(event.currentTarget, false, isWorkflowRunning)}
                role="menuitem"
              >
                Load Workflow
              </button>
              {executionHistory.length > 0 ? (
                <div
                  style={{
                    padding: '8px 10px',
                    color: 'var(--text-muted)',
                    fontSize: 11,
                    fontWeight: 700,
                    borderTop: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  Run history: {executionHistory.length} saved
                </div>
              ) : null}
            </div>
          ) : null}
          </div>
          {isWorkflowRunning ? (
            <button
              type="button"
              onClick={handleCancelRun}
              style={btnStyle('var(--accent-red)')}
              onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-red)', true)}
              onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-red)', false)}
              title="Cancel current workflow run"
            >
              Cancel
            </button>
          ) : null}
          <button
            onClick={handleClear}
            disabled={nodes.length === 0 || isWorkflowRunning}
            style={btnStyle('var(--accent-red)', nodes.length === 0 || isWorkflowRunning)}
            onMouseEnter={(event) => setButtonHover(event.currentTarget, 'var(--accent-red)', true, nodes.length === 0 || isWorkflowRunning)}
            onMouseLeave={(event) => setButtonHover(event.currentTarget, 'var(--accent-red)', false, nodes.length === 0 || isWorkflowRunning)}
            title="Clear all nodes"
          >
            🗑 Clear
          </button>
        </div>

        <input
          ref={workflowFileInputRef}
          type="file"
          accept="application/json,.json"
          onChange={handleWorkflowFileSelected}
          style={{ display: 'none' }}
        />

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onInit={(instance) => { reactFlowInstance.current = instance; }}
          onNodeClick={handleNodeClick}
          onNodeContextMenu={handleNodeContextMenu}
          onNodeDragStart={handleNodeDragStart}
          onEdgeClick={handleEdgeClick}
          onPaneClick={handlePaneClick}
          onPaneContextMenu={handlePaneContextMenu}
          onMoveStart={handleCanvasMoveStart}
          onMove={handleCanvasMove}
          onSelectionChange={handleSelectionChange}
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          defaultEdgeOptions={{ animated: true, type: 'typed' }}
          deleteKeyCode={['Delete', 'Backspace']}
          proOptions={{ hideAttribution: true }}
        >
          {suggestionMenu && suggestionMenuPosition ? (
            <NodeSuggestionMenu
              direction={suggestionMenu.direction}
              position={suggestionMenuPosition}
              existingNodes={compatibleExistingNodes}
              suggestions={compatibleSuggestions}
              getEdgePreview={getSuggestionEdgeData}
              onClose={() => setSuggestionMenu(null)}
              onConnectExisting={handleConnectExistingNode}
              onCreateNode={handleCreateSuggestedNode}
            />
          ) : null}
          <Background gap={20} size={1} color="var(--border-color)" />
          <Controls position="bottom-left" />
          {miniMapMinimized ? null : (
            <MiniMap
              position="bottom-right"
              nodeColor={() => 'var(--accent-blue)'}
              maskColor="rgba(26, 27, 46, 0.7)"
              style={{ width: 120, height: 90 }}
            />
          )}
        </ReactFlow>

        <button
          type="button"
          onClick={() => setMiniMapMinimized((value) => !value)}
          style={miniMapToggleStyle}
          title={miniMapMinimized ? 'Show minimap' : 'Hide minimap'}
          aria-label={miniMapMinimized ? 'Show minimap' : 'Hide minimap'}
        >
          {miniMapMinimized ? 'Map' : 'Hide Map'}
        </button>
      </div>

      {activeTaskTutorial ? (
        <TaskTutorialOverlay
          key={activeTaskTutorial.storageKey}
          config={activeTaskTutorial}
          nodes={nodes}
          viewportSignal={tutorialViewportSignal}
          doNotShowAgain={doNotShowTaskTutorial}
          onDoNotShowAgainChange={setDoNotShowTaskTutorial}
          onClose={handleCloseTaskTutorial}
        />
      ) : null}

      <NodeInspector />

      {showDeidentificationGuide && typeof document !== 'undefined'
        ? createPortal(
          <div
            style={guideOverlayStyle}
            onClick={handleCloseDeidentificationGuide}
            role="presentation"
          >
            <div
              style={guideModalStyle}
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="deidentification-guide-title"
            >
              <div style={{
                padding: '16px 18px',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                gap: 12,
                alignItems: 'flex-start',
              }}>
                <div>
                  <div
                    id="deidentification-guide-title"
                    style={{
                      color: 'var(--accent-green)',
                      fontSize: 14,
                      fontWeight: 900,
                      textTransform: 'uppercase',
                    }}
                  >
                    PHI Deidentification Guide
                  </div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 5, lineHeight: 1.5 }}>
                    This automated workflow prepares a copy of imaging data for research sharing or transfer.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleCloseDeidentificationGuide}
                  style={{
                    ...iconButtonStyle('var(--text-secondary)'),
                    minHeight: 30,
                    width: 30,
                    minWidth: 30,
                  }}
                  aria-label="Close deidentification guide"
                  title="Close"
                >
                  x
                </button>
              </div>

              <div style={{ padding: 18, overflowY: 'auto' }}>
                <div style={{
                  border: '1px solid color-mix(in srgb, var(--accent-green) 34%, var(--border-color))',
                  background: 'color-mix(in srgb, var(--accent-green) 8%, var(--bg-secondary))',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginBottom: 14,
                }}>
                  The source study is left untouched. The Deidentify node creates a new output path, blanks DICOM PHI fields, defaces supported 3D volumes when enabled, and writes to db/anonymization_audit.jsonl.
                </div>

                <div style={{ display: 'grid', gap: 10 }}>
                  {[
                    ['1', 'Load the study', 'Use Data Loader with a DICOM directory, uploaded ZIP, single DICOM file, or NIfTI volume.'],
                    ['2', 'Inspect before running', 'Metadata Before shows the fields currently present so the expert understands what will be removed.'],
                    ['3', 'Run the workflow', 'Deidentify creates a sanitized copy and logs the operation. Metadata After verifies the fields that remain visible.'],
                    ['4', 'Use the exported path', 'Export shows the sanitized artifact path for transfer, archiving, or downstream workflow use.'],
                  ].map(([number, title, text]) => (
                    <div
                      key={number}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '28px 1fr',
                        gap: 10,
                        alignItems: 'start',
                      }}
                    >
                      <div style={{
                        width: 26,
                        height: 26,
                        borderRadius: 6,
                        border: '1px solid color-mix(in srgb, var(--accent-green) 48%, var(--border-color))',
                        color: 'var(--accent-green)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 12,
                        fontWeight: 900,
                      }}>
                        {number}
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900 }}>
                          {title}
                        </div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.45, marginTop: 2 }}>
                          {text}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{
                padding: '12px 18px 16px',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
              }}>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  cursor: 'pointer',
                }}>
                  <input
                    type="checkbox"
                    checked={doNotShowDeidentificationGuide}
                    onChange={(event) => setDoNotShowDeidentificationGuide(event.target.checked)}
                  />
                  Do not show again
                </label>
                <button
                  type="button"
                  onClick={handleCloseDeidentificationGuide}
                  style={btnStyle('var(--accent-green)')}
                >
                  Got It
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )
        : null}

      <ResultModal
        isOpen={showDeidentificationResult}
        title="PHI Deidentification Complete"
        status="success"
        summary="The workflow finished and produced a sanitized study output."
        outputPath={deidentificationResultPath}
        expectation="The output should contain a deidentified copy of the source study. DICOM PHI fields are blanked, private tags are removed, supported NIfTI header text fields are cleared, and the audit trail records the operation."
        nextStep="Review Metadata Viewer (After Sanitization), then use the Export Sanitized Data node path for transfer, public-release preparation, or downstream workflow steps."
        doNotShowAgain={doNotShowDeidentificationResult}
        onDoNotShowAgainChange={setDoNotShowDeidentificationResult}
        onClose={handleCloseDeidentificationResult}
      />

      {showCrowdsourcingGuide && typeof document !== 'undefined'
        ? createPortal(
          <div
            style={guideOverlayStyle}
            onClick={handleCloseCrowdsourcingGuide}
            role="presentation"
          >
            <div
              style={guideModalStyle}
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="crowdsourcing-guide-title"
            >
              <div style={{
                padding: '16px 18px',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                gap: 12,
                alignItems: 'flex-start',
              }}>
                <div>
                  <div
                    id="crowdsourcing-guide-title"
                    style={{
                      color: 'var(--accent-green)',
                      fontSize: 14,
                      fontWeight: 900,
                      textTransform: 'uppercase',
                    }}
                  >
                    Crowdsourcing Task Guide
                  </div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 5, lineHeight: 1.5 }}>
                    Your assigned task is loaded automatically. Use this guide to complete it consistently.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleCloseCrowdsourcingGuide}
                  style={{
                    ...iconButtonStyle('var(--text-secondary)'),
                    minHeight: 30,
                    width: 30,
                    minWidth: 30,
                  }}
                  aria-label="Close crowdsourcing guide"
                  title="Close"
                >
                  x
                </button>
              </div>

              <div style={{ padding: 18, overflowY: 'auto' }}>
                <div style={{
                  border: '1px solid color-mix(in srgb, var(--accent-blue) 34%, var(--border-color))',
                  background: 'color-mix(in srgb, var(--accent-blue) 8%, var(--bg-secondary))',
                  borderRadius: 8,
                  padding: '10px 12px',
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginBottom: 14,
                }}>
                  Check the current patient in Your Assignment Tasks, annotate in Interactive Annotator, review suggested labels before accepting them, then submit the task when the patient is complete.
                </div>

                <div style={{ display: 'grid', gap: 10 }}>
                  {[
                    ['1', 'Confirm the loaded patient', 'Use Your Assignment Tasks on the canvas to verify the current patient and remaining queue.'],
                    ['2', 'Annotate and review', 'Work inside Interactive Annotator. Review AI segmentation and label suggestions before saving them.'],
                    ['3', 'Submit deliberately', 'Click Submit Task only when this assignment is ready. Next Task moves on without marking the patient complete.'],
                  ].map(([number, title, text]) => (
                    <div
                      key={number}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '28px 1fr',
                        gap: 10,
                        alignItems: 'start',
                      }}
                    >
                      <div style={{
                        width: 26,
                        height: 26,
                        borderRadius: 6,
                        border: '1px solid color-mix(in srgb, var(--accent-green) 48%, var(--border-color))',
                        color: 'var(--accent-green)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 12,
                        fontWeight: 900,
                      }}>
                        {number}
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 900 }}>
                          {title}
                        </div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.45, marginTop: 2 }}>
                          {text}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{
                padding: '12px 18px 16px',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
              }}>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                  cursor: 'pointer',
                }}>
                  <input
                    type="checkbox"
                    checked={doNotShowCrowdsourcingGuide}
                    onChange={(event) => setDoNotShowCrowdsourcingGuide(event.target.checked)}
                  />
                  Do not show again
                </label>
                <button
                  type="button"
                  onClick={handleCloseCrowdsourcingGuide}
                  style={btnStyle('var(--accent-green)')}
                >
                  Start Task
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )
        : null}
    </div>
  );
}
