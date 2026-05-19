/**
 * App.tsx — Main SegPro-Med Workflow Engine
 * ==========================================
 * React Flow canvas with node palette sidebar,
 * toolbar for workflow execution, and drag-and-drop node creation.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
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
import type { AnnotationShape, BaseNodeData, SliceAnnotationsMap } from './types/nodes';

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

const crowdsourcingBannerStyle: React.CSSProperties = {
  position: 'absolute',
  left: 12,
  bottom: 12,
  zIndex: 10,
  width: 420,
  maxWidth: 'calc(100% - 24px)',
  borderRadius: 8,
  border: '1px solid color-mix(in srgb, var(--accent-green) 35%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-green) 9%, var(--bg-secondary))',
  boxShadow: 'var(--shadow)',
  padding: 12,
};

const crowdsourcingMetaStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  color: 'var(--text-muted)',
  fontSize: 11,
  marginTop: 6,
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
  const [crowdsourcingSession, setCrowdsourcingSession] = useState<CrowdsourcingSession | null>(null);
  const [currentCrowdsourcingTask, setCurrentCrowdsourcingTask] = useState<api.CrowdsourcingTask | null>(null);
  const [crowdsourcingTaskQueue, setCrowdsourcingTaskQueue] = useState<api.CrowdsourcingTask[]>([]);
  const [crowdsourcingSubmitBusy, setCrowdsourcingSubmitBusy] = useState(false);
  const [crowdsourcingAutoRunToken, setCrowdsourcingAutoRunToken] = useState(0);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null);
  const workflowFileInputRef = useRef<HTMLInputElement | null>(null);
  const clipboardRef = useRef<ClipboardFragment | null>(null);
  const pasteOffsetRef = useRef(0);
  const activeRunControllerRef = useRef<AbortController | null>(null);
  const canUndo = historyPast.length > 0;
  const canRedo = historyFuture.length > 0;
  const currentWorkflowFingerprint = useMemo(
    () => getEditableWorkflowFingerprint(nodes, edges),
    [nodes, edges],
  );
  const hasUnsavedChanges = nodes.length > 0 && currentWorkflowFingerprint !== savedWorkflowFingerprint;
  const isWorkflowRunning = Boolean(activeRun);
  const selectedExecutionNodeId = selectedNodeIds[0] || selectedNodeId || undefined;

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

  const handleNodeDragStart = useCallback(() => {
    setNodeContextMenu(null);
    checkpointHistory();
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

  const handleCopySelectedNodes = useCallback(() => {
    const fragment = getSelectedFragment();
    if (!fragment) {
      setWorkflowNotice({
        type: 'info',
        message: 'Select one or more nodes before copying.',
      });
      return;
    }

    clipboardRef.current = cloneJson(fragment);
    pasteOffsetRef.current = 0;
    setWorkflowNotice({
      type: 'success',
      message: `Copied ${fragment.nodes.length} node${fragment.nodes.length === 1 ? '' : 's'}.`,
    });
  }, [getSelectedFragment, setWorkflowNotice]);

  const handlePasteNodes = useCallback(() => {
    const fragment = clipboardRef.current;
    if (!fragment || fragment.nodes.length === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Copy one or more nodes before pasting.',
      });
      return;
    }

    pasteOffsetRef.current += 36;
    pasteFragment(
      fragment,
      { x: pasteOffsetRef.current, y: pasteOffsetRef.current },
      `Pasted ${fragment.nodes.length} node${fragment.nodes.length === 1 ? '' : 's'}.`,
    );
  }, [pasteFragment, setWorkflowNotice]);

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
    pasteOffsetRef.current = 0;
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

      if (key === 'c') {
        event.preventDefault();
        handleCopySelectedNodes();
        return;
      }

      if (key === 'v') {
        event.preventDefault();
        handlePasteNodes();
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
    handleCopySelectedNodes,
    handleDuplicateSelectedNodes,
    handleRedoClick,
    handleUndoClick,
    handlePasteNodes,
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
    (task: api.CrowdsourcingTask, userId: string) => {
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
      const statusId = generateNodeId();
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
          data: {
            ...cloneDefaultData('interactiveAnnotator'),
            sourcePath: loadPath,
          },
        },
        {
          id: statusId,
          type: 'campaignStatus',
          position: { x: 790, y: 120 },
          data: {
            ...cloneDefaultData('campaignStatus'),
            campaignName: task.campaign_id,
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
      setCrowdsourcingAutoRunToken(Date.now());
      setWorkflowNotice({
        type: 'success',
        message: `Loaded assigned task ${task.patient_id} for ${userId}.`,
      });
    },
    [cloneDefaultData, replaceWorkflow, setSelectedNodeId, setWorkflowNotice],
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

      if (login.role === 'expert') {
        const firstTask = login.remaining_tasks[0];
        if (firstTask) {
          startCrowdsourcingTask(firstTask, login.user_id);
        } else {
          setCurrentCrowdsourcingTask(null);
          setWorkflowNotice({
            type: 'info',
            message: 'Login successful. No remaining crowdsourcing tasks are assigned to this expert.',
          });
        }
        return;
      }

      setWorkflowNotice({
        type: 'success',
        message: 'Admin login successful. Use Collaboration nodes to manage campaigns and assignments.',
      });
    },
    [setWorkflowNotice, startCrowdsourcingTask],
  );

  const handleOpenCrowdsourcingLogin = useCallback(() => {
    setShowCrowdsourcingLogin(true);
    setShowWorkflowMenu(false);
    setShowTemplatePanel(false);
    setNodeContextMenu(null);
  }, []);

  const handleCrowdsourcingLogout = useCallback(() => {
    setCrowdsourcingSession(null);
    setCurrentCrowdsourcingTask(null);
    setCrowdsourcingTaskQueue([]);
    setWorkflowNotice({
      type: 'info',
      message: 'Crowdsourcing session closed.',
    });
  }, [setWorkflowNotice]);

  const handleStartNextCrowdsourcingTask = useCallback(() => {
    if (!crowdsourcingSession) return;
    const nextTask = crowdsourcingTaskQueue.find((task) =>
      task.campaign_id !== currentCrowdsourcingTask?.campaign_id ||
      task.patient_id !== currentCrowdsourcingTask?.patient_id,
    ) || crowdsourcingTaskQueue[0];

    if (!nextTask) {
      setWorkflowNotice({
        type: 'info',
        message: 'No remaining crowdsourcing tasks are available.',
      });
      return;
    }

    startCrowdsourcingTask(nextTask, crowdsourcingSession.userId);
  }, [
    crowdsourcingSession,
    crowdsourcingTaskQueue,
    currentCrowdsourcingTask,
    setWorkflowNotice,
    startCrowdsourcingTask,
  ]);

  const handleSubmitCrowdsourcingTask = useCallback(async () => {
    if (!crowdsourcingSession || !currentCrowdsourcingTask) return;

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
        startCrowdsourcingTask(nextTask, crowdsourcingSession.userId);
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
    currentCrowdsourcingTask,
    nodes,
    setWorkflowNotice,
    startCrowdsourcingTask,
  ]);

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

  const handleCreateStarterWorkflow = useCallback(() => {
    const dataLoaderId = generateNodeId();
    const autoSegmentationId = generateNodeId();
    const annotatorId = generateNodeId();

    addNodesAndConnect(
      [
        {
          id: dataLoaderId,
          type: 'dataLoader',
          position: { x: 70, y: 90 },
          data: cloneDefaultData('dataLoader'),
        },
        {
          id: autoSegmentationId,
          type: 'autoSegmentation',
          position: { x: 430, y: 80 },
          data: cloneDefaultData('autoSegmentation'),
        },
        {
          id: annotatorId,
          type: 'interactiveAnnotator',
          position: { x: 790, y: 70 },
          data: cloneDefaultData('interactiveAnnotator'),
        },
      ],
      [
        { source: dataLoaderId, target: annotatorId, sourceHandle: null, targetHandle: null },
        { source: autoSegmentationId, target: annotatorId, sourceHandle: null, targetHandle: null },
      ],
    );
  }, [addNodesAndConnect, cloneDefaultData]);

  const handleApplyTemplate = useCallback(
    (template: WorkflowTemplate) => {
      const rightMostX = nodes.reduce(
        (max, node) => Math.max(max, node.position.x + (node.width ?? 320)),
        0,
      );
      const offset = nodes.length > 0
        ? { x: rightMostX + 120, y: 0 }
        : { x: 0, y: 0 };
      const instance = instantiateWorkflowTemplate(template, generateNodeId, offset);

      addNodesAndConnect(instance.nodes, instance.connections);
      setShowTemplatePanel(false);
      setShowValidationPanel(false);
      setWorkflowNotice({
        type: 'success',
        message: `${template.title} template added.`,
      });
    },
    [addNodesAndConnect, nodes, setWorkflowNotice],
  );

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
      <NodePalette />

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
            onCreateStarterWorkflow={handleCreateStarterWorkflow}
          />
        ) : null}

        {crowdsourcingSession && currentCrowdsourcingTask ? (
          <div style={crowdsourcingBannerStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'flex-start' }}>
              <div style={{ minWidth: 0 }}>
                <div
                  style={{
                    color: 'var(--accent-green)',
                    fontSize: 12,
                    fontWeight: 900,
                    textTransform: 'uppercase',
                    letterSpacing: 0.6,
                  }}
                >
                  Crowdsourcing Task
                </div>
                <div style={{ color: 'var(--text-primary)', fontSize: 14, fontWeight: 900, marginTop: 3 }}>
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
              <button
                type="button"
                onClick={handleCrowdsourcingLogout}
                style={btnStyle('var(--text-secondary)')}
                title="Close crowdsourcing session"
              >
                Logout
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
              <button
                type="button"
                onClick={() => { void handleSubmitCrowdsourcingTask(); }}
                disabled={crowdsourcingSubmitBusy || isWorkflowRunning}
                style={btnStyle('var(--accent-green)', crowdsourcingSubmitBusy || isWorkflowRunning)}
                title="Mark this assignment as completed in db/assignments.json"
              >
                {crowdsourcingSubmitBusy ? 'Submitting...' : 'Submit Task'}
              </button>
              <button
                type="button"
                onClick={handleStartNextCrowdsourcingTask}
                disabled={crowdsourcingTaskQueue.length === 0 || isWorkflowRunning}
                style={btnStyle('var(--accent-blue)', crowdsourcingTaskQueue.length === 0 || isWorkflowRunning)}
              >
                Next Task
              </button>
            </div>
          </div>
        ) : null}

        {/* Toolbar */}
        <div style={toolbarStyle}>
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
            onClick={
              !crowdsourcingSession
                ? handleOpenCrowdsourcingLogin
                : crowdsourcingTaskQueue.length > 0
                  ? handleStartNextCrowdsourcingTask
                  : handleCrowdsourcingLogout
            }
            disabled={isWorkflowRunning}
            style={btnStyle(
              crowdsourcingSession ? 'var(--accent-green)' : 'var(--accent-purple)',
              isWorkflowRunning,
            )}
            onMouseEnter={(event) => setButtonHover(
              event.currentTarget,
              crowdsourcingSession ? 'var(--accent-green)' : 'var(--accent-purple)',
              true,
              isWorkflowRunning,
            )}
            onMouseLeave={(event) => setButtonHover(
              event.currentTarget,
              crowdsourcingSession ? 'var(--accent-green)' : 'var(--accent-purple)',
              false,
              isWorkflowRunning,
            )}
            title={
              !crowdsourcingSession
                ? 'Login to assigned crowdsourcing tasks'
                : crowdsourcingTaskQueue.length > 0
                  ? 'Load the next assigned task'
                  : 'No remaining tasks. Close crowdsourcing session.'
            }
          >
            {crowdsourcingSession
              ? `Crowd: ${crowdsourcingSession.userId}`
              : 'Crowdsourcing'}
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
          <MiniMap
            position="bottom-right"
            nodeColor={() => 'var(--accent-blue)'}
            maskColor="rgba(26, 27, 46, 0.7)"
          />
        </ReactFlow>
      </div>

      <NodeInspector />
    </div>
  );
}
