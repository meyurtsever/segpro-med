/**
 * App.tsx — Main SegPro-Med Workflow Engine
 * ==========================================
 * React Flow canvas with node palette sidebar,
 * toolbar for workflow execution, and drag-and-drop node creation.
 */

import { useCallback, useMemo, useRef, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import useWorkflowStore, { generateNodeId } from './store/workflowStore';
import { nodeTypes } from './nodes';
import NodePalette from './panels/NodePalette';
import { executeWorkflow } from './engine/executor';
import { createIsValidConnection } from './engine/compatibility';

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

export default function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    updateNodeData,
    clearWorkflow,
  } = useWorkflowStore();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null);

  // --- Connection validation (compatibility matrix) ---
  const isValidConnection = useMemo(
    () => createIsValidConnection(nodes, edges),
    [nodes, edges],
  );

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
    if (nodes.length === 0) return;

    // Reset all nodes to idle first
    for (const node of nodes) {
      updateNodeData(node.id, { status: 'idle', error: undefined });
    }

    try {
      await executeWorkflow(nodes, edges, updateNodeData);
    } catch (err) {
      console.error('Workflow execution error:', err);
    }
  }, [nodes, edges, updateNodeData]);

  const handleClear = useCallback(() => {
    if (nodes.length > 0 && !confirm('Clear all nodes and edges?')) return;
    clearWorkflow();
  }, [nodes, clearWorkflow]);

  return (
    <div style={appStyle}>
      {/* Left sidebar — Node Palette */}
      <NodePalette />

      {/* Main canvas */}
      <div style={canvasContainerStyle}>
        {/* Toolbar */}
        <div style={toolbarStyle}>
          <button
            onClick={handleRun}
            style={btnStyle('var(--accent-green)')}
            title="Execute the workflow"
          >
            ▶ Run Workflow
          </button>
          <button
            onClick={handleClear}
            style={btnStyle('var(--accent-red)')}
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
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          fitView
          defaultEdgeOptions={{ animated: true }}
          deleteKeyCode={['Delete', 'Backspace']}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={20} size={1} color="var(--border-color)" />
          <Controls position="bottom-left" />
          <MiniMap
            position="bottom-right"
            nodeColor={() => 'var(--accent-blue)'}
            maskColor="rgba(26, 27, 46, 0.7)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}
