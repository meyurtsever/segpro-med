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

interface GraphSnapshot {
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
}

interface WorkflowState {
  // Graph state
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
  historyPast: GraphSnapshot[];
  historyFuture: GraphSnapshot[];
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
  insertGraphFragment: (
    nodes: Node<BaseNodeData>[],
    edges: Edge[],
  ) => void;
  updateNodeData: (nodeId: string, data: Partial<BaseNodeData>) => void;
  removeNode: (nodeId: string) => void;
  removeEdge: (edgeId: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setSelectedEdgeId: (edgeId: string | null) => void;
  setWorkflowNotice: (notice: WorkflowState['workflowNotice']) => void;

  // Actions — workflow management
  clearWorkflow: () => void;
  replaceWorkflow: (nodes: Node<BaseNodeData>[], edges: Edge[]) => void;
  checkpointHistory: () => void;
  undoWorkflow: () => void;
  redoWorkflow: () => void;
}

let nodeCounter = 0;
export const generateNodeId = () => `node_${++nodeCounter}`;
const HISTORY_LIMIT = 50;

function cloneGraph(nodes: Node<BaseNodeData>[], edges: Edge[]): GraphSnapshot {
  return JSON.parse(JSON.stringify({ nodes, edges })) as GraphSnapshot;
}

function graphFingerprint(snapshot: GraphSnapshot) {
  return JSON.stringify({
    nodes: snapshot.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: node.data,
    })),
    edges: snapshot.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle,
      targetHandle: edge.targetHandle,
      type: edge.type,
      data: edge.data,
    })),
  });
}

function pushHistory(state: WorkflowState) {
  const snapshot = cloneGraph(state.nodes, state.edges);
  const last = state.historyPast[state.historyPast.length - 1];

  if (last && graphFingerprint(last) === graphFingerprint(snapshot)) {
    return state.historyPast;
  }

  return [...state.historyPast, snapshot].slice(-HISTORY_LIMIT);
}

function syncNodeCounter(nodes: Node<BaseNodeData>[]) {
  const maxFromIds = nodes.reduce((max, node) => {
    const match = /^node_(\d+)$/.exec(node.id);
    return match ? Math.max(max, Number(match[1])) : max;
  }, 0);

  nodeCounter = Math.max(nodeCounter, maxFromIds, nodes.length);
}

const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  historyPast: [],
  historyFuture: [],
  selectedNodeId: null,
  selectedEdgeId: null,
  workflowNotice: null,

  onNodesChange: (changes) => {
    const shouldTrack = changes.some((change) =>
      change.type !== 'select' &&
      change.type !== 'position' &&
      change.type !== 'dimensions',
    );
    const removedIds = new Set(
      changes
        .filter((change) => change.type === 'remove')
        .map((change) => change.id),
    );
    const selectedNodeId = get().selectedNodeId;
    const state = get();

    set({
      nodes: applyNodeChanges(changes, state.nodes) as Node<BaseNodeData>[],
      historyPast: shouldTrack ? pushHistory(state) : state.historyPast,
      historyFuture: shouldTrack ? [] : state.historyFuture,
      selectedNodeId: selectedNodeId && removedIds.has(selectedNodeId)
        ? null
        : selectedNodeId,
    });
  },

  onEdgesChange: (changes) => {
    const shouldTrack = changes.some((change) => change.type !== 'select');
    const removedIds = new Set(
      changes
        .filter((change) => change.type === 'remove')
        .map((change) => change.id),
    );
    const selectedEdgeId = get().selectedEdgeId;
    const state = get();

    set({
      edges: applyEdgeChanges(changes, state.edges),
      historyPast: shouldTrack ? pushHistory(state) : state.historyPast,
      historyFuture: shouldTrack ? [] : state.historyFuture,
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

      // MetadataViewer targets: propagate already loaded metadata when available
      if (targetNode.type === 'metadataViewer') {
        if (sourceNode.type === 'dataLoader') {
          patch = {
            sessionId: sourceData.sessionId as string | undefined,
            sourcePath: (sourceData.filePath as string | undefined) ||
              (sourceData.path as string | undefined),
            fileType: sourceData.fileType as string | undefined,
            volumeShape: sourceData.volumeShape as number[] | undefined,
            metadata: sourceData.metadata as Record<string, unknown> | undefined,
          };
        }
        if (sourceNode.type === 'formatConverter') {
          patch = {
            sourcePath: sourceData.outputPath as string | undefined,
          };
        }
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
      historyPast: pushHistory(get()),
      historyFuture: [],
      workflowNotice: {
        type: 'success',
        message: 'Connection added.',
      },
    });
  },

  addNode: (node) => {
    set({
      nodes: [...get().nodes, node],
      historyPast: pushHistory(get()),
      historyFuture: [],
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
      historyPast: pushHistory(get()),
      historyFuture: [],
      selectedNodeId: nodesToAdd[0]?.id ?? get().selectedNodeId,
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },

  insertGraphFragment: (nodesToAdd, edgesToAdd) => {
    if (nodesToAdd.length === 0) return;

    const existingNodes = get().nodes.map((node) => ({
      ...node,
      selected: false,
    }));
    const allNodes = [
      ...existingNodes,
      ...nodesToAdd.map((node) => ({
        ...node,
        selected: true,
      })),
    ];
    let nextEdges = get().edges.map((edge) => ({
      ...edge,
      selected: false,
    }));

    for (const edge of edgesToAdd) {
      const result = validateConnection(edge, allNodes, nextEdges);
      if (!result.valid) continue;

      const sourceNode = allNodes.find((node) => node.id === edge.source);
      const targetNode = allNodes.find((node) => node.id === edge.target);
      nextEdges = addEdge({
        ...edge,
        type: 'typed',
        animated: true,
        selected: false,
        data: sourceNode && targetNode
          ? getConnectionEdgeData(sourceNode.type, targetNode.type)
          : edge.data,
      }, nextEdges);
    }

    set({
      nodes: allNodes,
      edges: nextEdges,
      historyPast: pushHistory(get()),
      historyFuture: [],
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
      historyPast: pushHistory(get()),
      historyFuture: [],
    });
  },

  removeEdge: (edgeId) => {
    set({
      edges: get().edges.filter((edge) => edge.id !== edgeId),
      selectedEdgeId: get().selectedEdgeId === edgeId ? null : get().selectedEdgeId,
      historyPast: pushHistory(get()),
      historyFuture: [],
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
      historyPast: get().nodes.length || get().edges.length ? pushHistory(get()) : get().historyPast,
      historyFuture: [],
      selectedNodeId: null,
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },

  replaceWorkflow: (nodes, edges) => {
    syncNodeCounter(nodes);
    set({
      nodes,
      edges,
      historyPast: pushHistory(get()),
      historyFuture: [],
      selectedNodeId: nodes[0]?.id ?? null,
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },

  checkpointHistory: () => {
    const state = get();
    if (state.nodes.length === 0 && state.edges.length === 0) return;
    set({
      historyPast: pushHistory(state),
      historyFuture: [],
    });
  },

  undoWorkflow: () => {
    const { historyPast, nodes, edges } = get();
    const previous = historyPast[historyPast.length - 1];
    if (!previous) return;

    syncNodeCounter(previous.nodes);
    set({
      nodes: previous.nodes,
      edges: previous.edges,
      historyPast: historyPast.slice(0, -1),
      historyFuture: [cloneGraph(nodes, edges), ...get().historyFuture].slice(0, HISTORY_LIMIT),
      selectedNodeId: previous.nodes[0]?.id ?? null,
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },

  redoWorkflow: () => {
    const { historyFuture, nodes, edges } = get();
    const next = historyFuture[0];
    if (!next) return;

    syncNodeCounter(next.nodes);
    set({
      nodes: next.nodes,
      edges: next.edges,
      historyPast: [...get().historyPast, cloneGraph(nodes, edges)].slice(-HISTORY_LIMIT),
      historyFuture: historyFuture.slice(1),
      selectedNodeId: next.nodes[0]?.id ?? null,
      selectedEdgeId: null,
      workflowNotice: null,
    });
  },
}));

export default useWorkflowStore;
