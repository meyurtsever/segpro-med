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
  ViewportPortal,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import useWorkflowStore, { generateNodeId } from './store/workflowStore';
import { nodePaletteItems, nodeTypes } from './nodes';
import NodePalette from './panels/NodePalette';
import { executeWorkflow } from './engine/executor';
import { createIsValidConnection, getAllowedSources, getAllowedTargets } from './engine/compatibility';
import NodeInspector from './panels/NodeInspector';

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

const suggestionMenuStyle: React.CSSProperties = {
  position: 'absolute',
  zIndex: 50,
  width: SUGGESTION_MENU_WIDTH,
  pointerEvents: 'all',
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-color)',
  borderRadius: 7,
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const suggestionHeaderStyle: React.CSSProperties = {
  padding: '7px 9px',
  borderBottom: '1px solid var(--border-color)',
  color: 'var(--text-secondary)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0.7,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 8,
};

const suggestionCloseButtonStyle: React.CSSProperties = {
  width: 18,
  height: 18,
  borderRadius: 4,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  fontSize: 12,
  lineHeight: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

const suggestionItemStyle: React.CSSProperties = {
  width: '100%',
  border: 'none',
  background: 'transparent',
  color: 'var(--text-primary)',
  padding: '7px 9px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'flex-start',
  gap: 7,
  textAlign: 'left',
};

const suggestionSectionTitleStyle: React.CSSProperties = {
  padding: '7px 9px 3px',
  color: 'var(--text-muted)',
  fontSize: 9,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0.7,
};

const suggestionDividerStyle: React.CSSProperties = {
  height: 1,
  background: 'var(--border-color)',
  margin: '4px 0',
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
    updateNodeData,
    clearWorkflow,
    setSelectedNodeId,
    setWorkflowNotice,
  } = useWorkflowStore();
  const [suggestionMenu, setSuggestionMenu] = useState<SuggestionMenuState | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null);

  // --- Connection validation (compatibility matrix) ---
  const isValidConnection = useMemo(
    () => createIsValidConnection(nodes, edges),
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

  // --- Workflow execution ---
  const handleRun = useCallback(async () => {
    if (nodes.length === 0) {
      setWorkflowNotice({
        type: 'info',
        message: 'Add at least one node before running the workflow.',
      });
      return;
    }

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
  }, [nodes, edges, updateNodeData, setWorkflowNotice]);

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

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setSuggestionMenu(null);
  }, [setSelectedNodeId]);

  const handleSelectionChange = useCallback(
    ({ nodes: selectedNodes }: { nodes: Array<{ id: string }> }) => {
      setSelectedNodeId(selectedNodes[0]?.id ?? null);
    },
    [setSelectedNodeId],
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

  const hasCompatibleExistingNodes = compatibleExistingNodes.length > 0;
  const hasCompatibleSuggestions = compatibleSuggestions.length > 0;

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
          onPaneClick={handlePaneClick}
          onSelectionChange={handleSelectionChange}
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          fitView
          defaultEdgeOptions={{ animated: true }}
          deleteKeyCode={['Delete', 'Backspace']}
          proOptions={{ hideAttribution: true }}
        >
          {suggestionMenu && suggestionMenuPosition ? (
            <ViewportPortal>
              <div
                style={{
                  ...suggestionMenuStyle,
                  transform: `translate(${suggestionMenuPosition.x}px, ${suggestionMenuPosition.y}px)`,
                }}
                className="nodrag nopan nowheel"
                onPointerDown={(event) => event.stopPropagation()}
                onMouseDown={(event) => event.stopPropagation()}
                onClick={(event) => event.stopPropagation()}
                onWheel={(event) => event.stopPropagation()}
              >
                <div style={suggestionHeaderStyle}>
                  <span>
                    {suggestionMenu.direction === 'output'
                      ? 'Compatible next nodes'
                      : 'Compatible previous nodes'}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSuggestionMenu(null)}
                    style={suggestionCloseButtonStyle}
                    title="Close suggestions"
                    aria-label="Close suggestions"
                  >
                    x
                  </button>
                </div>

                {hasCompatibleExistingNodes ? (
                  <>
                    <div style={suggestionSectionTitleStyle}>Connect existing</div>
                    {compatibleExistingNodes.map((node) => {
                      const item = nodePaletteItems.find((paletteNode) => paletteNode.type === node.type);
                      return (
                        <button
                          key={node.id}
                          type="button"
                          onClick={() => handleConnectExistingNode(node.id)}
                          style={suggestionItemStyle}
                          onMouseEnter={(event) => {
                            event.currentTarget.style.background = 'var(--bg-tertiary)';
                          }}
                          onMouseLeave={(event) => {
                            event.currentTarget.style.background = 'transparent';
                          }}
                        >
                          <span style={{ fontSize: 15, lineHeight: 1 }}>{item?.icon || '*'}</span>
                          <span style={{ minWidth: 0, fontSize: 11, fontWeight: 700 }}>
                            {String(node.data.label || item?.label || node.type)}
                          </span>
                        </button>
                      );
                    })}
                  </>
                ) : null}

                {hasCompatibleExistingNodes && hasCompatibleSuggestions ? (
                  <div style={suggestionDividerStyle} />
                ) : null}
                <div style={suggestionSectionTitleStyle}>Create new</div>
                {!hasCompatibleSuggestions ? (
                  <div style={{ padding: 10, color: 'var(--text-muted)', fontSize: 12 }}>
                    No compatible node types.
                  </div>
                ) : (
                  compatibleSuggestions.map((item) => (
                    <button
                      key={item.type}
                      type="button"
                      onClick={() => handleCreateSuggestedNode(item.type, item.defaultData)}
                      style={suggestionItemStyle}
                      onMouseEnter={(event) => {
                        event.currentTarget.style.background = 'var(--bg-tertiary)';
                      }}
                      onMouseLeave={(event) => {
                        event.currentTarget.style.background = 'transparent';
                      }}
                    >
                      <span style={{ fontSize: 15, lineHeight: 1 }}>{item.icon}</span>
                      <span style={{ minWidth: 0 }}>
                        <span style={{ display: 'block', fontSize: 11, fontWeight: 700 }}>
                          {item.label}
                        </span>
                        <span
                          style={{
                            display: 'block',
                            fontSize: 9,
                            color: 'var(--text-muted)',
                            marginTop: 1,
                            lineHeight: 1.35,
                          }}
                        >
                          {item.description}
                        </span>
                      </span>
                    </button>
                  ))
                )}
              </div>
            </ViewportPortal>
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
