/**
 * App.tsx — Main SegPro-Med Workflow Engine
 * ==========================================
 * React Flow canvas with node palette sidebar,
 * toolbar for workflow execution, and drag-and-drop node creation.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import useWorkflowStore, { generateNodeId } from './store/workflowStore';
import { nodePaletteItems, nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import NodePalette from './panels/NodePalette';
import { executeWorkflow } from './engine/executor';
import { createIsValidConnection, getAllowedSources, getAllowedTargets } from './engine/compatibility';
import { getConnectionEdgeData } from './engine/nodeContracts';
import NodeInspector from './panels/NodeInspector';
import WorkflowValidationPanel from './panels/WorkflowValidationPanel';
import NodeSuggestionMenu from './components/NodeSuggestionMenu';
import CanvasEmptyState from './components/CanvasEmptyState';
import { validateWorkflow, type WorkflowIssue } from './engine/workflowValidation';

interface SuggestionMenuState {
  sourceNodeId: string;
  sourceNodeType: string;
  direction: 'input' | 'output';
}

const SUGGESTION_MENU_WIDTH = 226;
const SUGGESTION_MENU_GAP = 28;

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
};

const btnStyle = (color: string): React.CSSProperties => ({
  padding: '8px 16px',
  borderRadius: 6,
  border: `1px solid ${color}60`,
  background: `${color}18`,
  color,
  fontWeight: 600,
  fontSize: 12,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  transition: 'all 0.15s ease',
});

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
    workflowNotice,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    addNodesAndConnect,
    updateNodeData,
    clearWorkflow,
    setSelectedNodeId,
    setSelectedEdgeId,
    setWorkflowNotice,
  } = useWorkflowStore();
  const [suggestionMenu, setSuggestionMenu] = useState<SuggestionMenuState | null>(null);
  const [showValidationPanel, setShowValidationPanel] = useState(false);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null);

  // --- Connection validation (compatibility matrix) ---
  const isValidConnection = useMemo(
    () => createIsValidConnection(nodes, edges),
    [nodes, edges],
  );

  const validationIssues = useMemo(
    () => validateWorkflow(nodes, edges),
    [nodes, edges],
  );

  const blockingIssues = useMemo(
    () => validationIssues.filter((issue) => issue.severity === 'error'),
    [validationIssues],
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

  // --- Workflow execution ---
  const handleRun = useCallback(async () => {
    if (nodes.length === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Add at least one node before running the workflow.',
      });
      return;
    }

    if (blockingIssues.length > 0) {
      const firstIssue = blockingIssues[0];
      if (firstIssue.nodeId) {
        setSelectedNodeId(firstIssue.nodeId);
      }
      setShowValidationPanel(true);
      setWorkflowNotice({
        type: 'warning',
        message: firstIssue.message,
      });
      return;
    }

    setShowValidationPanel(false);

    // Reset all nodes to idle first
    for (const node of nodes) {
      updateNodeData(node.id, { status: 'idle', error: undefined });
    }

    try {
      await executeWorkflow(nodes, edges, updateNodeData);
      setWorkflowNotice({
        type: 'success',
        message: 'Workflow completed.',
      });
    } catch (err) {
      setWorkflowNotice({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [blockingIssues, nodes, edges, updateNodeData, setSelectedNodeId, setWorkflowNotice]);

  const handleClear = useCallback(() => {
    if (nodes.length > 0 && !confirm('Clear all nodes and edges?')) return;
    clearWorkflow();
  }, [nodes, clearWorkflow]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId],
  );

  const handleEdgeClick = useCallback(
    (_: React.MouseEvent, edge: { id: string }) => {
      setSelectedEdgeId(edge.id);
      setSuggestionMenu(null);
    },
    [setSelectedEdgeId],
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setSuggestionMenu(null);
  }, [setSelectedEdgeId, setSelectedNodeId]);

  const handleValidationIssueClick = useCallback(
    (issue: WorkflowIssue) => {
      if (issue.nodeId) {
        setSelectedNodeId(issue.nodeId);
      }
    },
    [setSelectedNodeId],
  );

  const handleSelectionChange = useCallback(
    ({
      nodes: selectedNodes,
      edges: selectedEdges,
    }: {
      nodes: Array<{ id: string }>;
      edges: Array<{ id: string }>;
    }) => {
      setSelectedNodeId(selectedNodes[0]?.id ?? null);
      setSelectedEdgeId(selectedNodes.length === 0 ? selectedEdges[0]?.id ?? null : null);
    },
    [setSelectedEdgeId, setSelectedNodeId],
  );

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
        { source: dataLoaderId, target: autoSegmentationId, sourceHandle: null, targetHandle: null },
        { source: autoSegmentationId, target: annotatorId, sourceHandle: null, targetHandle: null },
      ],
    );
  }, [addNodesAndConnect, cloneDefaultData]);

  const getSuggestionEdgeData = useCallback(
    (candidateType: string | undefined) => {
      if (!suggestionMenu) return undefined;
      return suggestionMenu.direction === 'output'
        ? getConnectionEdgeData(suggestionMenu.sourceNodeType, candidateType)
        : getConnectionEdgeData(candidateType, suggestionMenu.sourceNodeType);
    },
    [suggestionMenu],
  );

  return (
    <div style={appStyle}>
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

        {nodes.length === 0 ? (
          <CanvasEmptyState
            onCreateDataLoader={handleCreateDataLoader}
            onCreateStarterWorkflow={handleCreateStarterWorkflow}
          />
        ) : null}

        {/* Toolbar */}
        <div style={toolbarStyle}>
          <button
            onClick={handleRun}
            disabled={nodes.length === 0}
            style={{
              ...btnStyle('var(--accent-green)'),
              opacity: nodes.length === 0 ? 0.45 : 1,
              cursor: nodes.length === 0 ? 'not-allowed' : 'pointer',
            }}
            title="Execute the workflow"
          >
            ▶ Run Workflow
          </button>
          <button
            onClick={handleClear}
            disabled={nodes.length === 0}
            style={{
              ...btnStyle('var(--accent-red)'),
              opacity: nodes.length === 0 ? 0.45 : 1,
              cursor: nodes.length === 0 ? 'not-allowed' : 'pointer',
            }}
            title="Clear all nodes"
          >
            🗑 Clear
          </button>
        </div>

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onInit={(instance) => { reactFlowInstance.current = instance; }}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          onPaneClick={handlePaneClick}
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
