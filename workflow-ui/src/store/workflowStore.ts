/**
 * Zustand store for React Flow workflow state.
 * Manages nodes, edges, and execution state.
 */

import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react';

import type { BaseNodeData } from '../types/nodes';
import { validateConnection } from '../engine/compatibility';

interface WorkflowState {
  // Graph state
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  workflowNotice: {
    type: 'info' | 'success' | 'warning' | 'error';
    message: string;
  } | null;

  // Actions — React Flow handlers
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;

  // Actions — node management
  addNode: (node: Node<BaseNodeData>) => void;
  updateNodeData: (nodeId: string, data: Partial<BaseNodeData>) => void;
  removeNode: (nodeId: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setWorkflowNotice: (notice: WorkflowState['workflowNotice']) => void;

  // Actions — workflow management
  clearWorkflow: () => void;
}

let nodeCounter = 0;
export const generateNodeId = () => `node_${++nodeCounter}`;

const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  workflowNotice: null,

  onNodesChange: (changes) => {
    const removedIds = new Set(
      changes
        .filter((change) => change.type === 'remove')
        .map((change) => change.id),
    );
    const selectedNodeId = get().selectedNodeId;

    set({
      nodes: applyNodeChanges(changes, get().nodes) as Node<BaseNodeData>[],
      selectedNodeId: selectedNodeId && removedIds.has(selectedNodeId)
        ? null
        : selectedNodeId,
    });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection) => {
    // Validate connection compatibility before adding
    const { nodes, edges } = get();
    const result = validateConnection(connection, nodes, edges);
    if (!result.valid) {
      set({
        workflowNotice: {
          type: 'warning',
          message: result.reason || 'These nodes cannot be connected.',
        },
      });
      return;
    }

    const newEdges = addEdge({ ...connection, animated: true }, edges);

    // Auto-propagate data from upstream node on new connection
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);
    let newNodes = nodes;

    if (sourceNode && targetNode) {
      const sourceData = sourceNode.data as Record<string, unknown>;
      let patch: Record<string, unknown> | undefined;

      // FormatConverter targets: propagate inputPath
      if (targetNode.type === 'formatConverter') {
        let propagatedPath: string | undefined;
        if (sourceNode.type === 'dataLoader') {
          propagatedPath = sourceData.path as string | undefined;
        } else if (sourceNode.type === 'formatConverter') {
          propagatedPath = sourceData.outputPath as string | undefined;
        }
        if (propagatedPath) {
          patch = { inputPath: propagatedPath };
        }
      }

      // SliceViewer targets: propagate sessionId or path
      if (targetNode.type === 'sliceViewer') {
        if (sourceNode.type === 'dataLoader' && sourceData.sessionId) {
          patch = { sessionId: sourceData.sessionId as string };
        }
        if (sourceNode.type === 'interactiveAnnotator' && sourceData.sessionId) {
          patch = { sessionId: sourceData.sessionId as string };
        }
        // FormatConverter → SliceViewer: no immediate propagation;
        // the executor will load the file and set sessionId at run time
      }

      // InteractiveAnnotator targets: propagate sessionId
      if (targetNode.type === 'interactiveAnnotator') {
        if (sourceNode.type === 'dataLoader' && sourceData.sessionId) {
          patch = { sessionId: sourceData.sessionId as string };
        }
        if (sourceNode.type === 'autoSegmentation' && sourceData.sessionId) {
          patch = { sessionId: sourceData.sessionId as string };
        }
        // FormatConverter → InteractiveAnnotator: handled at execution time
      }

      // AutoSegmentation targets: propagate sessionId
      if (targetNode.type === 'autoSegmentation') {
        if (sourceNode.type === 'dataLoader' && sourceData.sessionId) {
          patch = { sessionId: sourceData.sessionId as string };
        }
        // FormatConverter → AutoSegmentation: handled at execution time
      }

      if (patch) {
        newNodes = nodes.map((n) =>
          n.id === connection.target
            ? { ...n, data: { ...n.data, ...patch } }
            : n,
        );
      }
    }

    set({
      edges: newEdges,
      nodes: newNodes,
      workflowNotice: {
        type: 'success',
        message: 'Connection added.',
      },
    });
  },

  addNode: (node) => {
    set({
      nodes: [...get().nodes, node],
      selectedNodeId: node.id,
      workflowNotice: null,
    });
  },

  updateNodeData: (nodeId, data) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    });
  },

  removeNode: (nodeId) => {
    set({
      nodes: get().nodes.filter((n) => n.id !== nodeId),
      edges: get().edges.filter(
        (e) => e.source !== nodeId && e.target !== nodeId,
      ),
      selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
    });
  },

  setSelectedNodeId: (nodeId) => {
    if (get().selectedNodeId === nodeId) return;
    set({ selectedNodeId: nodeId });
  },

  setWorkflowNotice: (notice) => {
    const current = get().workflowNotice;
    if (
      current?.type === notice?.type &&
      current?.message === notice?.message
    ) {
      return;
    }
    set({ workflowNotice: notice });
  },

  clearWorkflow: () => {
    nodeCounter = 0;
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      workflowNotice: null,
    });
  },
}));

export default useWorkflowStore;
