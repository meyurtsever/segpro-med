/**
 * TypeScript types for all workflow nodes and their data.
 */

/** Status of a node's execution */
export type NodeStatus = 'idle' | 'running' | 'success' | 'error' | 'cancelled' | 'skipped';

/** Base data shared by all node types */
export interface BaseNodeData {
  label: string;
  status: NodeStatus;
  error?: string;
  [key: string]: unknown;
}

/** DataLoaderNode — loads DICOM/NIfTI/MAT from path or upload */
export interface DataLoaderNodeData extends BaseNodeData {
  path: string;
  sessionId?: string;
  fileType?: string;
  volumeShape?: number[];
  metadata?: Record<string, unknown>;
  /** Optional root directory for the patient search (ICD-10 / keyword mode) */
  searchRoot?: string;
}

/** FormatConverterNode — converts between medical image formats */
export interface FormatConverterNodeData extends BaseNodeData {
  inputPath: string;
  outputPath: string;
  conversionType: string;
  axis?: number;
  compress?: boolean;
  outputSizeBytes?: number;
  previewSessionId?: string;
  previewFileType?: string;
  previewImageBase64?: string;
  previewTotalSlices?: number;
  previewVolumeShape?: number[];
  previewMetadata?: Record<string, unknown>;
}

/** MetadataViewerNode - inspects metadata from a loaded session or file path */
export interface MetadataViewerNodeData extends BaseNodeData {
  sessionId?: string;
  sourcePath?: string;
  fileType?: string;
  volumeShape?: number[];
  metadata?: Record<string, unknown>;
  filterText?: string;
}

/** Coordinate selected on a slice image */
export interface SliceCoordinate {
  /** X position in image pixels (0-based) */
  x: number;
  /** Y position in image pixels (0-based) */
  y: number;
  /** Slice index at time of selection */
  sliceIndex: number;
  /** View plane at time of selection */
  view: 'axial' | 'sagittal' | 'coronal';
}

/** SliceViewerNode — inline slice preview */
export interface SliceViewerNodeData extends BaseNodeData {
  sessionId: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  totalSlices: number;
  imageBase64?: string;
  /** Zoom level (1 = 100%) */
  zoom?: number;
  /** Selected coordinate on the slice */
  selectedCoord?: SliceCoordinate;
  /** Volume shape from upstream metadata [D, H, W] */
  volumeShape?: number[];
  /** Upstream metadata for auto-view-plane detection */
  metadata?: Record<string, unknown>;
  /** Path to the NIfTI segmentation file (e.g. Untitled.nii alongside FLAIR) */
  segPath?: string;
  /** Whether the segmentation overlay is currently visible */
  showOverlay?: boolean;
}

// ---------------------------------------------------------------------------
// Interactive Annotator Node
// ---------------------------------------------------------------------------

/** A single point in image-space (pixel coordinates, persisted) */
export interface ImagePoint {
  x: number;
  y: number;
}

/** Polygon annotation — ordered array of vertices */
export interface PolygonAnnotation {
  type: 'polygon';
  /** Vertices in image-space pixel coordinates (persisted) */
  points: ImagePoint[];
  label: string;
  color: string;
}

/** Circle annotation — center + radius in image-space */
export interface CircleAnnotation {
  type: 'circle';
  /** Center in image-space pixel coordinates */
  centerX: number;
  centerY: number;
  /** Radius in image-space pixels */
  radius: number;
  label: string;
  color: string;
}

/** Freehand annotation — dense array of points */
export interface FreehandAnnotation {
  type: 'freehand';
  /** Ordered path points in image-space pixel coordinates */
  points: ImagePoint[];
  label: string;
  color: string;
}

/** Point annotation — a single coordinate marker */
export interface PointAnnotation {
  type: 'point';
  /** Location in image-space pixel coordinates */
  x: number;
  y: number;
  label: string;
  color: string;
}

/** Rectangle annotation — axis-aligned bounding box in image-space */
export interface RectAnnotation {
  type: 'rect';
  /** Top-left corner in image-space pixel coordinates */
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color: string;
}

/** Union of all annotation shapes */
export type AnnotationShape =
  | PolygonAnnotation
  | CircleAnnotation
  | FreehandAnnotation
  | PointAnnotation
  | RectAnnotation;

/** Active drawing tool */
export type AnnotationTool = 'rect' | 'polygon' | 'circle' | 'freehand' | 'point' | 'pan' | 'eraser';

/** Per-slice annotation storage map: { [sliceIndex: number]: AnnotationShape[] } */
export type SliceAnnotationsMap = Record<number, AnnotationShape[]>;

export type SegmentationRunMode = 'single' | 'range' | 'wholeVolume';

export interface SegmentationSliceResult {
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  shapes: AnnotationShape[];
  count: number;
  rawMaskCount: number;
  filteredCount: number;
  elapsedSeconds: number;
}

export interface PromptBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PromptPoint extends ImagePoint {
  label: 0 | 1;
}

export interface SegmentationPrompt {
  sessionId: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  points: PromptPoint[];
  boxes: PromptBox[];
  source: 'interactiveAnnotator' | 'annotationLoad' | 'manual';
}

/** Reusable SAM2 segmentation output passed between workflow nodes */
export interface SegmentationResult {
  sessionId: string;
  runMode?: SegmentationRunMode;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  shapes: AnnotationShape[];
  sliceResults?: SegmentationSliceResult[];
  segmentedSliceCount?: number;
  count: number;
  rawMaskCount: number;
  filteredCount: number;
  configUsed: string;
  elapsedSeconds: number;
  message: string;
}

export type VlmModelId = 'medgemma' | 'smolvlm' | 'med-r1';
export type VlmModality = 'MRI' | 'CT' | 'MG';
export type VoicePromptIntent = 'describe' | 'anomaly' | 'both' | 'custom';

export interface VoicePrompt {
  text: string;
  intent: VoicePromptIntent;
  promptKey: string;
  identifyAnomalies: boolean;
  describeSlice: boolean;
  source: 'typed' | 'dictation' | 'upload' | 'audio_path' | 'manual';
}

export interface VlmPromptPreset {
  key: string;
  title: string;
  description: string;
  prompt: string;
  modality: VlmModality;
  source: string;
  parameters: Record<string, unknown>;
}

export interface VlmAnalysisResult {
  model: VlmModelId;
  modelLabel: string;
  modality: VlmModality;
  promptKey: string;
  promptTitle: string;
  promptUsed: string;
  sessionId: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  text: string;
  labels: string[];
  elapsedSeconds: number;
}

export interface LabelSuggestionResult {
  model: VlmModelId;
  modality: VlmModality;
  labels: string[];
  rawText: string;
  promptKey: string;
  promptTitle: string;
  elapsedSeconds: number;
}

export interface CampaignProgress {
  totalPatients: number;
  assignedPatients: number;
  completed: number;
  reviewed: number;
  unassignedPatients: number;
}

export interface ExpertAssignment {
  expertId: string;
  assignedPatients: string[];
  completedPatients: string[];
  pendingPatients: string[];
}

export interface CampaignInfo {
  name: string;
  datasetPath: string;
  description?: string;
  createdAt?: string | null;
  totalPatients: number;
  patients: string[];
  progress: CampaignProgress;
  unassignedPatients: string[];
  assignments: ExpertAssignment[];
}

export interface PatientAssignmentResult {
  campaignName: string;
  expertId: string;
  assignedPatients: string[];
  assignmentCount: number;
  message: string;
}

/** InteractiveAnnotatorNode — canvas-based annotation on slices */
export interface InteractiveAnnotatorNodeData extends BaseNodeData {
  sessionId: string;
  sourcePath?: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  totalSlices: number;
  imageBase64?: string;
  /** All annotations on the current slice (persisted in image-space) */
  annotations: AnnotationShape[];
  /** Persistent per-slice annotation storage (survives slice navigation) */
  sliceAnnotationsMap: SliceAnnotationsMap;
  /** Currently active tool */
  activeTool: AnnotationTool;
  /** Zoom level (1 = 100%) */
  zoom: number;
  /** Whether annotation labels are visible on canvas */
  showLabels: boolean;
  /** Volume shape from upstream metadata [D, H, W] */
  volumeShape?: number[];
  /** Upstream metadata for auto-view-plane detection */
  metadata?: Record<string, unknown>;
  /** Optional AI segmentation result loaded as draft annotations */
  segmentationResult?: SegmentationResult;
  /** Latest point/box prompt selected for prompt-driven SAM2 */
  segmentationPrompt?: SegmentationPrompt;
  /** Suggested semantic labels from the VLM Label Suggester */
  labelSuggestions?: string[];
  labelSuggestionResult?: LabelSuggestionResult;
}

// ---------------------------------------------------------------------------
// Segmentation Profile Node
// ---------------------------------------------------------------------------

/** A segmentation config profile from the backend */
export interface SegmentationConfig {
  name: string;
  description: string;
  points_per_side: number;
  pred_iou_thresh: number;
  min_mask_region_area: number;
}

/** AutoSegmentationNode - configuration-only SAM2 profile provider */
export interface AutoSegmentationNodeData extends BaseNodeData {
  /** Currently selected config profile name */
  configName: string;
  /** Available configs fetched from backend */
  availableConfigs: SegmentationConfig[];
  /** Upstream metadata for view detection */
  metadata?: Record<string, unknown>;
  /** Volume shape from upstream [D, H, W] */
  volumeShape?: number[];
}

/** MedSAM2SegmenterNode - executes SAM2 automatic segmentation */
export interface MedSAM2SegmenterNodeData extends BaseNodeData {
  /** Session inherited from upstream DataLoader/FormatConverter */
  sessionId: string;
  /** Segmentation execution scope */
  runMode?: SegmentationRunMode;
  /** Slice to segment */
  sliceIndex: number;
  /** Start slice for range mode */
  sliceStart?: number;
  /** End slice for range mode */
  sliceEnd?: number;
  /** Slice interval for range and whole-volume modes */
  sliceStep?: number;
  /** View plane for segmentation */
  view: 'axial' | 'sagittal' | 'coronal';
  /** Selected SAM2 config profile */
  configName: string;
  /** Prompt mode for point/box-guided segmentation */
  promptMode?: 'auto' | 'prompt';
  /** Prompt source used when promptMode is prompt */
  segmentationPrompt?: SegmentationPrompt;
  /** Available configs fetched from backend */
  availableConfigs: SegmentationConfig[];
  /** Latest segmentation output */
  segmentationResult?: SegmentationResult;
  /** Convenience copy of generated shapes */
  segmentationShapes?: AnnotationShape[];
  /** Upstream metadata for view detection */
  metadata?: Record<string, unknown>;
  /** Volume shape from upstream [D, H, W] */
  volumeShape?: number[];
}

export interface AnnotationRecord {
  userId: string;
  studyPath: string;
  annotationCount: number;
  data?: Record<string, unknown>;
  exportPath?: string;
  message?: string;
}

export interface AnnotationStoreNodeData extends BaseNodeData {
  userId: string;
  studyPath: string;
  annotationType: string;
  annotationCount?: number;
  annotationRecord?: AnnotationRecord;
}

export interface AnnotationLoadNodeData extends BaseNodeData {
  userId: string;
  studyPath: string;
  annotationCount?: number;
  annotations: AnnotationShape[];
  sliceAnnotationsMap: SliceAnnotationsMap;
  annotationRecord?: AnnotationRecord;
}

export interface ExportNodeData extends BaseNodeData {
  userId: string;
  studyPath: string;
  exportFormat: 'json';
  outputPath?: string;
  annotationCount?: number;
  annotationRecord?: AnnotationRecord;
}

export interface VlmNodeData extends BaseNodeData {
  sessionId: string;
  sourcePath?: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  modality: VlmModality;
  promptKey: string;
  customPrompt?: string;
  maxTokens?: number;
  includeReasoning?: boolean;
  useOverlay?: boolean;
  availablePrompts: VlmPromptPreset[];
  annotations?: AnnotationShape[];
  sliceAnnotationsMap?: SliceAnnotationsMap;
  voicePrompt?: VoicePrompt;
  vlmResult?: VlmAnalysisResult;
}

export interface LabelSuggesterNodeData extends BaseNodeData {
  sessionId: string;
  sourcePath?: string;
  sliceIndex: number;
  view: 'axial' | 'sagittal' | 'coronal';
  model: VlmModelId;
  modality: VlmModality;
  promptKey: string;
  customPrompt?: string;
  maxLabels: number;
  useOverlay?: boolean;
  availablePrompts: VlmPromptPreset[];
  annotations?: AnnotationShape[];
  sliceAnnotationsMap?: SliceAnnotationsMap;
  voicePrompt?: VoicePrompt;
  currentLabels?: string[];
  labelSuggestions?: string[];
  labelSuggestionResult?: LabelSuggestionResult;
  vlmResult?: VlmAnalysisResult;
}

export interface VoiceInputNodeData extends BaseNodeData {
  transcript: string;
  audioPath?: string;
  audioFileName?: string;
  autoDetectIntent?: boolean;
  intent: VoicePromptIntent;
  promptKey: string;
  voicePrompt?: VoicePrompt;
}

export interface CampaignSetupNodeData extends BaseNodeData {
  campaignName: string;
  datasetPath: string;
  description?: string;
  totalPatients?: number;
  patients?: string[];
  campaign?: CampaignInfo;
}

export interface PatientAssignNodeData extends BaseNodeData {
  campaignName: string;
  expertId: string;
  assignmentMode: 'selected' | 'allUnassigned';
  patientIdsText: string;
  availableExperts?: string[];
  unassignedPatients?: string[];
  assignmentCount?: number;
  assignmentResult?: PatientAssignmentResult;
  campaign?: CampaignInfo;
}

export interface CampaignStatusNodeData extends BaseNodeData {
  campaignName: string;
  campaign?: CampaignInfo;
}

// ---------------------------------------------------------------------------
// Palette & Registry types
// ---------------------------------------------------------------------------

/** Node category for palette grouping */
export interface NodeCategory {
  id: string;
  label: string;
  icon: string;
  color: string;
  nodes: NodeTypeInfo[];
}

/** Info for a node type in the palette */
export interface NodeTypeInfo {
  type: string;
  label: string;
  description: string;
  icon: string;
  category: string;
  defaultData: BaseNodeData;
}
