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

/** InteractiveAnnotatorNode — canvas-based annotation on slices */
export interface InteractiveAnnotatorNodeData extends BaseNodeData {
  sessionId: string;
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
}

// ---------------------------------------------------------------------------
// Auto Segmentation Node
// ---------------------------------------------------------------------------

/** A segmentation config profile from the backend */
export interface SegmentationConfig {
  name: string;
  description: string;
  points_per_side: number;
  pred_iou_thresh: number;
  min_mask_region_area: number;
}

/** AutoSegmentationNode — configures and triggers SAM2 auto-segmentation */
export interface AutoSegmentationNodeData extends BaseNodeData {
  /** Session inherited from upstream DataLoader/FormatConverter */
  sessionId: string;
  /** Currently selected config profile name */
  configName: string;
  /** Available configs fetched from backend */
  availableConfigs: SegmentationConfig[];
  /** Upstream metadata for view detection */
  metadata?: Record<string, unknown>;
  /** Volume shape from upstream [D, H, W] */
  volumeShape?: number[];
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
