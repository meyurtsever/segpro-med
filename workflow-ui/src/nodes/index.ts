/**
 * Node Type Registry
 * ==================
 * Central registry mapping node type strings to React components.
 * React Flow uses this to render custom nodes.
 */

import type { NodeTypes } from '@xyflow/react';
import DataLoaderNode from './DataLoaderNode';
import FormatConverterNode from './FormatConverterNode';
import MetadataViewerNode from './MetadataViewerNode';
import SliceViewerNode from './SliceViewerNode';
import InteractiveAnnotatorNode from './InteractiveAnnotatorNode';
import AutoSegmentationNode from './AutoSegmentationNode';
import MedSAM2SegmenterNode from './MedSAM2SegmenterNode';
import AnnotationStoreNode from './AnnotationStoreNode';
import AnnotationLoadNode from './AnnotationLoadNode';
import ExportNode from './ExportNode';
import VoiceInputNode from './VoiceInputNode';
import {
  MedGemmaWorkflowNode,
  MedR1WorkflowNode,
  SmolVlmWorkflowNode,
} from './VlmModelNode';
import LabelSuggesterNode from './LabelSuggesterNode';
import CampaignSetupNode from './CampaignSetupNode';
import PatientAssignNode from './PatientAssignNode';
import CampaignStatusNode from './CampaignStatusNode';

export { nodePaletteItems } from '../engine/nodeContracts';

export const nodeTypes: NodeTypes = {
  dataLoader: DataLoaderNode,
  formatConverter: FormatConverterNode,
  metadataViewer: MetadataViewerNode,
  sliceViewer: SliceViewerNode,
  interactiveAnnotator: InteractiveAnnotatorNode,
  autoSegmentation: AutoSegmentationNode,
  medsam2Segmenter: MedSAM2SegmenterNode,
  annotationStore: AnnotationStoreNode,
  annotationLoad: AnnotationLoadNode,
  exportNode: ExportNode,
  voiceInput: VoiceInputNode,
  medgemmaNode: MedGemmaWorkflowNode,
  smolvlmNode: SmolVlmWorkflowNode,
  medR1Node: MedR1WorkflowNode,
  labelSuggester: LabelSuggesterNode,
  campaignSetup: CampaignSetupNode,
  patientAssign: PatientAssignNode,
  campaignStatus: CampaignStatusNode,
};
