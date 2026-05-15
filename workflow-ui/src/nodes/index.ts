/**
 * Node Type Registry
 * ==================
 * Central registry mapping node type strings to React components.
 * React Flow uses this to render custom nodes.
 */

import type { NodeTypes } from '@xyflow/react';
import DataLoaderNode from './DataLoaderNode';
import FormatConverterNode from './FormatConverterNode';
import SliceViewerNode from './SliceViewerNode';
import InteractiveAnnotatorNode from './InteractiveAnnotatorNode';
import AutoSegmentationNode from './AutoSegmentationNode';

export { nodePaletteItems } from '../engine/nodeContracts';

export const nodeTypes: NodeTypes = {
  dataLoader: DataLoaderNode,
  formatConverter: FormatConverterNode,
  sliceViewer: SliceViewerNode,
  interactiveAnnotator: InteractiveAnnotatorNode,
  autoSegmentation: AutoSegmentationNode,
};
