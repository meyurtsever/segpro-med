/**
 * Node Type Registry
 * ===================
 * Central registry mapping node type strings → React components.
 * React Flow uses this to render custom nodes.
 */

import type { NodeTypes } from '@xyflow/react';
import DataLoaderNode from './DataLoaderNode';
import FormatConverterNode from './FormatConverterNode';
import SliceViewerNode from './SliceViewerNode';
import InteractiveAnnotatorNode from './InteractiveAnnotatorNode';
import AutoSegmentationNode from './AutoSegmentationNode';

export const nodeTypes: NodeTypes = {
  dataLoader: DataLoaderNode,
  formatConverter: FormatConverterNode,
  sliceViewer: SliceViewerNode,
  interactiveAnnotator: InteractiveAnnotatorNode,
  autoSegmentation: AutoSegmentationNode,
};

/**
 * Node palette definitions — what appears in the sidebar.
 * Each entry describes a draggable node type.
 */
export const nodePaletteItems = [
  {
    type: 'dataLoader',
    label: 'Data Loader',
    icon: '📂',
    category: 'Data I/O',
    categoryColor: 'var(--accent-blue)',
    description: 'Load DICOM / NIfTI / MAT from local path',
    defaultData: {
      label: 'Data Loader',
      status: 'idle' as const,
      path: '',
    },
  },
  {
    type: 'formatConverter',
    label: 'Format Converter',
    icon: '🔄',
    category: 'Data I/O',
    categoryColor: 'var(--accent-orange)',
    description: 'Convert between DICOM, NIfTI, MAT, PNG',
    defaultData: {
      label: 'Format Converter',
      status: 'idle' as const,
      inputPath: '',
      outputPath: '',
      conversionType: '',
    },
  },
  {
    type: 'sliceViewer',
    label: 'Slice Viewer',
    icon: '🖼️',
    category: 'Visualization',
    categoryColor: 'var(--accent-purple)',
    description: 'View image slices with navigation controls',
    defaultData: {
      label: 'Slice Viewer',
      status: 'idle' as const,
      sessionId: '',
      sliceIndex: 0,
      view: 'axial' as const,
      totalSlices: 0,
      zoom: 1,
    },
  },
  {
    type: 'interactiveAnnotator',
    label: 'Interactive Annotator',
    icon: '✏️',
    category: 'Annotation',
    categoryColor: 'var(--accent-green)',
    description: 'Draw polygons, circles, freehand paths & coordinate markers on slices',
    defaultData: {
      label: 'Interactive Annotator',
      status: 'idle' as const,
      sessionId: '',
      sliceIndex: 0,
      view: 'axial' as const,
      totalSlices: 0,
      annotations: [],
      sliceAnnotationsMap: {},
      activeTool: 'rect' as const,
      zoom: 1,
      showLabels: true,
    },
  },
  {
    type: 'autoSegmentation',
    label: 'Auto Segmentation',
    icon: '🔬',
    category: 'Segmentation',
    categoryColor: 'var(--accent-purple)',
    description: 'SAM2 automatic ROI detection — connect to Interactive Annotator',
    defaultData: {
      label: 'Auto Segmentation',
      status: 'idle' as const,
      sessionId: '',
      configName: 'fast',
      availableConfigs: [],
    },
  },
];
