/**
 * Zustand store for React Flow workflow state.
 * Manages nodes, edges, and execution state.
 */

import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type Connection,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react';

import type { BaseNodeData } from '../types/nodes';
import { validateConnection } from '../engine/compatibility';
import { getConnectionEdgeData } from '../engine/nodeContracts';

interface WorkflowState {
  // Graph state
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
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
  addNodesAndConnect: (
    nodes: Node<BaseNodeData>[],
    connections: Connection[],
  ) => void;
  updateNodeData: (nodeId: string, data: Partial<BaseNodeData>) => void;
  removeNode: (nodeId: string) => void;
  removeEdge: (edgeId: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setSelectedEdgeId: (edgeId: string | null) => void;
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
  selectedEdgeId: null,
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
    const removedIds = new Set(
      changes
        .filter((change) => change.type === 'remove')
        .map((change) => change.id),
    );
    const selectedEdgeId = get().selectedEdgeId;

    set({
      edges: applyEdgeChanges(changes, get().edges),
      selectedEdgeId: selectedEdgeId && removedIds.has(selectedEdgeId)
        ? null
        : selectedEdgeId,
    });
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

    // Auto-propagate data from upstream node on new connection
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);
    const edgeData = sourceNode && targetNode
      ? getConnectionEdgeData(sourceNode.type, targetNode.type)
      : undefined;
    const newEdges = addEdge({
      ...connection,
      type: 'typed',
      animated: true,
      data: edgeData,
    }, edges);
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
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },

  addNodesAndConnect: (nodesToAdd, connections) => {
    const existingNodes = get().nodes;
    const allNodes = [...existingNodes, ...nodesToAdd];
    let nextEdges = get().edges;

    for (const connection of connections) {
      const result = validateConnection(connection, allNodes, nextEdges);
      if (!result.valid) continue;

      const sourceNode = allNodes.find((node) => node.id === connection.source);
      const targetNode = allNodes.find((node) => node.id === connection.target);
      nextEdges = addEdge({
        ...connection,
        type: 'typed',
        animated: true,
        data: sourceNode && targetNode
          ? getConnectionEdgeData(sourceNode.type, targetNode.type)
          : undefined,
      }, nextEdges);
    }

    set({
      nodes: allNodes,
      edges: nextEdges,
      selectedNodeId: nodesToAdd[0]?.id ?? get().selectedNodeId,
      selectedEdgeId: null,
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
    const selectedEdgeId = get().selectedEdgeId;
    const selectedEdge = get().edges.find((edge) => edge.id === selectedEdgeId);
    set({
      nodes: get().nodes.filter((n) => n.id !== nodeId),
      edges: get().edges.filter(
        (e) => e.source !== nodeId && e.target !== nodeId,
      ),
      selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
      selectedEdgeId: selectedEdge &&
        (selectedEdge.source === nodeId || selectedEdge.target === nodeId)
        ? null
        : selectedEdgeId,
    });
  },

  removeEdge: (edgeId) => {
    set({
      edges: get().edges.filter((edge) => edge.id !== edgeId),
      selectedEdgeId: get().selectedEdgeId === edgeId ? null : get().selectedEdgeId,
    });
  },

  setSelectedNodeId: (nodeId) => {
    if (get().selectedNodeId === nodeId) return;
    set({ selectedNodeId: nodeId, selectedEdgeId: nodeId ? null : get().selectedEdgeId });
  },

  setSelectedEdgeId: (edgeId) => {
    if (get().selectedEdgeId === edgeId) return;
    set({ selectedEdgeId: edgeId, selectedNodeId: edgeId ? null : get().selectedNodeId });
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
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },
}));

export default useWorkflowStore;
