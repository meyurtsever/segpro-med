/**
 * Typed HTTP client for the SegPro-Med FastAPI backend.
 * All endpoints per §5.3 of WORKFLOW_TRANSFORMATION_ROADMAP.md
 */

const API_BASE = 'http://localhost:8000/api/v1';

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

async function request<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}: ${res.statusText}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Data I/O  (/api/v1/data/*)
// ---------------------------------------------------------------------------

export interface LoadDataResponse {
  session_id: string;
  file_path: string;
  file_type: string;
  volume_shape: number[];
  metadata: Record<string, unknown>;
  message: string;
}

export interface SliceResponse {
  session_id: string;
  slice_index: number;
  view: string;
  total_slices: number;
  image_base64: string;
  width: number;
  height: number;
}

export interface MetadataResponse {
  session_id: string;
  file_path: string;
  file_type: string;
  volume_shape: number[];
  metadata: Record<string, unknown>;
}

/** Load data from a local filesystem path */
export async function loadDataFromPath(
  path: string,
  signal?: AbortSignal,
): Promise<LoadDataResponse> {
  return request<LoadDataResponse>(`${API_BASE}/data/load`, {
    method: 'POST',
    signal,
    body: JSON.stringify({ path }),
  });
}

/** Upload a file (for browser-based uploads) */
export async function uploadData(file: File): Promise<LoadDataResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/data/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

/** Get a single slice as base64 PNG */
export async function getSlice(
  sessionId: string,
  sliceIndex: number = 0,
  view: string = 'axial',
  segPath?: string,
  signal?: AbortSignal,
): Promise<SliceResponse> {
  const params = new URLSearchParams({
    slice: String(sliceIndex),
    view,
  });
  if (segPath) params.set('seg_path', segPath);
  return request<SliceResponse>(`${API_BASE}/data/slice/${sessionId}?${params}`, {
    signal,
  });
}

/** Get volume metadata */
export async function getMetadata(
  sessionId: string,
  signal?: AbortSignal,
): Promise<MetadataResponse> {
  return request<MetadataResponse>(`${API_BASE}/data/metadata/${sessionId}`, {
    signal,
  });
}

// ---------------------------------------------------------------------------
// Format Conversion  (/api/v1/convert/*)
// ---------------------------------------------------------------------------

export interface ConvertResponse {
  input_path: string;
  output_path: string;
  conversion_type: string;
  success: boolean;
  message: string;
  output_size_bytes?: number;
  metadata?: Record<string, unknown>;
}

export interface ConvertTypesResponse {
  conversions: Array<{
    type: string;
    input: string;
    output: string;
    description: string;
  }>;
}

export interface ConversionOptions {
  axis?: number;
  compress?: boolean;
}

/** List supported conversion types */
export async function getConversionTypes(): Promise<ConvertTypesResponse> {
  return request<ConvertTypesResponse>(`${API_BASE}/convert/types`);
}

/** Run a format conversion */
export async function runConversion(
  inputPath: string,
  conversionType: string,
  outputPath?: string,
  options: ConversionOptions = {},
  signal?: AbortSignal,
): Promise<ConvertResponse> {
  return request<ConvertResponse>(`${API_BASE}/convert/run`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      input_path: inputPath,
      conversion_type: conversionType,
      output_path: outputPath,
      axis: options.axis,
      compress: options.compress,
    }),
  });
}

// ---------------------------------------------------------------------------
// Filesystem Browsing  (/api/v1/fs/*)
// ---------------------------------------------------------------------------

export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  extension: string;
}

export interface BrowseResponse {
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
}

export interface NativePathDialogResponse {
  path: string | null;
  cancelled: boolean;
  mode: 'file' | 'directory';
}

/** Browse a filesystem directory (for file picker) */
export async function browsePath(path: string): Promise<BrowseResponse> {
  return request<BrowseResponse>(
    `${API_BASE}/fs/browse?path=${encodeURIComponent(path)}`,
  );
}

/** Open a native OS file/folder selector on the local API host */
export async function openNativePathDialog(
  mode: 'file' | 'directory',
  initialPath?: string,
): Promise<NativePathDialogResponse> {
  const params = new URLSearchParams({ mode });
  if (initialPath) params.set('initial_path', initialPath);
  return request<NativePathDialogResponse>(`${API_BASE}/fs/dialog?${params}`);
}

// ---------------------------------------------------------------------------
// Patient Search  (/api/v1/patients/*)
// ---------------------------------------------------------------------------

export interface PatientSearchResult {
  display_name: string;
  path: string;
  anomaly_class: string;
  icd10_code: string;
  icd10_description: string;
  segmentation_path: string;
}

export interface PatientSearchResponse {
  results: PatientSearchResult[];
  total: number;
}

/**
 * Search patients by ICD-10 code or keyword.
 * Supports ICD-10 codes ('D18.02'), clinical aliases ('cavernoma'),
 * and folder/patient name substrings.
 *
 * @param q        Search query (min 1 char)
 * @param rootDir  Root directory of the structured dataset (optional;
 *                 defaults to the path configured in utils/patient_retrieval.py)
 */
export async function searchPatients(
  q: string,
  rootDir?: string,
): Promise<PatientSearchResponse> {
  const params = new URLSearchParams({ q });
  if (rootDir) params.set('root_dir', rootDir);
  return request<PatientSearchResponse>(`${API_BASE}/patients/search?${params}`);
}

export interface PatientRootResponse {
  root_directory: string;
  directory_exists: boolean;
  cache_initialized: boolean;
  total_patients: number | null;
}

/** Get the current root directory and status of the patient retrieval singleton */
export async function getPatientRoot(): Promise<PatientRootResponse> {
  return request<PatientRootResponse>(`${API_BASE}/patients/root`);
}

// ---------------------------------------------------------------------------
// Segmentation  (/api/v1/segmentation/*)
// ---------------------------------------------------------------------------

export interface SegmentationPoint {
  x: number;
  y: number;
}

export interface SegmentationShape {
  type: string;
  points: SegmentationPoint[];
  label: string;
  color: string;
}

export interface AutoSegmentResponse {
  shapes: SegmentationShape[];
  count: number;
  raw_mask_count: number;
  config_used: string;
  elapsed_seconds: number;
  message: string;
}

export interface PromptPoint {
  x: number;
  y: number;
  label?: 0 | 1;
}

export interface PromptBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SegmentationConfigInfo {
  name: string;
  description: string;
  points_per_side: number;
  pred_iou_thresh: number;
  min_mask_region_area: number;
}

export interface SegmentationConfigsResponse {
  configs: SegmentationConfigInfo[];
}

/** Run SAM2 automatic segmentation on a single slice */
export async function autoSegment(
  sessionId: string,
  sliceIndex: number = 0,
  view: string = 'axial',
  configName: string = 'fast',
  signal?: AbortSignal,
): Promise<AutoSegmentResponse> {
  return request<AutoSegmentResponse>(`${API_BASE}/segmentation/auto`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      session_id: sessionId,
      slice_index: sliceIndex,
      view,
      config_name: configName,
    }),
  });
}

/** Run SAM2 from point and/or box prompts on a single slice */
export async function promptSegment(
  sessionId: string,
  sliceIndex: number = 0,
  view: string = 'axial',
  points: PromptPoint[] = [],
  boxes: PromptBox[] = [],
  configName: string = 'fast',
  signal?: AbortSignal,
): Promise<AutoSegmentResponse> {
  return request<AutoSegmentResponse>(`${API_BASE}/segmentation/prompt`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      session_id: sessionId,
      slice_index: sliceIndex,
      view,
      config_name: configName,
      points,
      boxes,
    }),
  });
}

/** List available segmentation config profiles */
export async function getSegmentationConfigs(): Promise<SegmentationConfigsResponse> {
  return request<SegmentationConfigsResponse>(`${API_BASE}/segmentation/configs`);
}

// ---------------------------------------------------------------------------
// VLM  (/api/v1/vlm/*)
// ---------------------------------------------------------------------------

export type VlmModelId = 'medgemma' | 'smolvlm' | 'med-r1';
export type VlmModality = 'MRI' | 'CT' | 'MG';

export interface VlmModelInfo {
  id: VlmModelId;
  label: string;
  description: string;
  strengths: string[];
}

export interface VlmModelsResponse {
  models: VlmModelInfo[];
}

export interface VlmPromptPreset {
  key: string;
  title: string;
  description: string;
  prompt: string;
  modality: VlmModality;
  source: 'gradio' | 'workflow' | string;
  parameters: Record<string, unknown>;
}

export interface VlmPromptsResponse {
  modality: VlmModality;
  prompts: VlmPromptPreset[];
}

export interface VlmPoint {
  x: number;
  y: number;
}

export interface VlmAnnotationShape {
  type: string;
  points?: VlmPoint[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  radius?: number;
  label?: string;
  color?: string;
}

export interface VlmAnalysisRequest {
  session_id: string;
  slice_index?: number;
  view?: string;
  model?: VlmModelId;
  modality?: VlmModality;
  prompt_key?: string;
  custom_prompt?: string;
  max_tokens?: number;
  include_reasoning?: boolean;
  annotations?: VlmAnnotationShape[];
  use_overlay?: boolean;
}

export interface VlmAnalysisResponse {
  success: boolean;
  model: VlmModelId;
  model_label: string;
  modality: VlmModality;
  prompt_key: string;
  prompt_title: string;
  prompt_used: string;
  session_id: string;
  slice_index: number;
  view: string;
  text: string;
  labels: string[];
  elapsed_seconds: number;
}

export interface VlmLabelSuggestRequest extends VlmAnalysisRequest {
  current_labels?: string[];
  max_labels?: number;
}

export type VlmLabelSuggestResponse = VlmAnalysisResponse;

export async function getVlmModels(): Promise<VlmModelsResponse> {
  return request<VlmModelsResponse>(`${API_BASE}/vlm/models`);
}

export async function getVlmPrompts(
  modality: VlmModality = 'MRI',
  signal?: AbortSignal,
): Promise<VlmPromptsResponse> {
  const params = new URLSearchParams({ modality });
  return request<VlmPromptsResponse>(`${API_BASE}/vlm/prompts?${params}`, { signal });
}

export async function runVlmAnalysis(
  requestBody: VlmAnalysisRequest,
  signal?: AbortSignal,
): Promise<VlmAnalysisResponse> {
  return request<VlmAnalysisResponse>(`${API_BASE}/vlm/analyze`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

export async function suggestVlmLabels(
  requestBody: VlmLabelSuggestRequest,
  signal?: AbortSignal,
): Promise<VlmLabelSuggestResponse> {
  return request<VlmLabelSuggestResponse>(`${API_BASE}/vlm/labels`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

// ---------------------------------------------------------------------------
// Voice  (/api/v1/voice/*)
// ---------------------------------------------------------------------------

export type VoiceIntent = 'describe' | 'anomaly' | 'both' | 'custom';

export interface VoicePromptResponse {
  success: boolean;
  transcript: string;
  intent: VoiceIntent;
  prompt_key: string;
  identify_anomalies: boolean;
  describe_slice: boolean;
  source: 'typed' | 'dictation' | 'upload' | 'audio_path' | string;
  message: string;
}

export async function inferVoiceIntent(
  transcript: string,
  signal?: AbortSignal,
): Promise<VoicePromptResponse> {
  return request<VoicePromptResponse>(`${API_BASE}/voice/intent`, {
    method: 'POST',
    signal,
    body: JSON.stringify({ transcript }),
  });
}

export async function transcribeVoicePath(
  audioPath: string,
  signal?: AbortSignal,
): Promise<VoicePromptResponse> {
  return request<VoicePromptResponse>(`${API_BASE}/voice/transcribe-path`, {
    method: 'POST',
    signal,
    body: JSON.stringify({ audio_path: audioPath }),
  });
}

export async function transcribeVoiceFile(
  file: File,
  signal?: AbortSignal,
): Promise<VoicePromptResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/voice/transcribe`, {
    method: 'POST',
    signal,
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Voice transcription failed: ${res.statusText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Collaboration  (/api/v1/collaboration/*)
// ---------------------------------------------------------------------------

export interface CollaborationProgress {
  total_patients: number;
  assigned_patients: number;
  completed: number;
  reviewed: number;
  unassigned_patients: number;
}

export interface CollaborationExpertAssignment {
  expert_id: string;
  assigned_patients: string[];
  completed_patients: string[];
  pending_patients: string[];
}

export interface CollaborationCampaign {
  name: string;
  dataset_path: string;
  description: string;
  created_at: string | null;
  total_patients: number;
  patients: string[];
  progress: CollaborationProgress;
  unassigned_patients: string[];
  assignments: CollaborationExpertAssignment[];
}

export interface CollaborationScanResponse {
  dataset_path: string;
  total_patients: number;
  patients: string[];
}

export interface CollaborationCreateResponse {
  success: boolean;
  created: boolean;
  campaign: CollaborationCampaign;
  message: string;
}

export interface CollaborationListResponse {
  campaigns: CollaborationCampaign[];
}

export interface CollaborationAssignmentOptionsResponse {
  campaign: CollaborationCampaign;
  experts: string[];
  unassigned_patients: string[];
}

export interface CollaborationAssignRequest {
  campaign_name: string;
  expert_id: string;
  patient_ids?: string[];
  assignment_mode?: 'selected' | 'allUnassigned';
}

export interface CollaborationAssignResponse {
  success: boolean;
  campaign_name: string;
  expert_id: string;
  assigned_patients: string[];
  assignment_count: number;
  campaign: CollaborationCampaign;
  message: string;
}

export async function scanCollaborationDataset(
  datasetPath: string,
  signal?: AbortSignal,
): Promise<CollaborationScanResponse> {
  return request<CollaborationScanResponse>(`${API_BASE}/collaboration/scan`, {
    method: 'POST',
    signal,
    body: JSON.stringify({ dataset_path: datasetPath }),
  });
}

export async function createCollaborationCampaign(
  campaignName: string,
  datasetPath: string,
  description: string = '',
  signal?: AbortSignal,
): Promise<CollaborationCreateResponse> {
  return request<CollaborationCreateResponse>(`${API_BASE}/collaboration/campaigns`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      campaign_name: campaignName,
      dataset_path: datasetPath,
      description,
    }),
  });
}

export async function getCollaborationCampaign(
  campaignName: string,
  signal?: AbortSignal,
): Promise<CollaborationCampaign> {
  return request<CollaborationCampaign>(
    `${API_BASE}/collaboration/campaigns/${encodeURIComponent(campaignName)}`,
    { signal },
  );
}

export async function getCollaborationCampaigns(
  signal?: AbortSignal,
): Promise<CollaborationListResponse> {
  return request<CollaborationListResponse>(`${API_BASE}/collaboration/campaigns`, { signal });
}

export async function getCollaborationAssignmentOptions(
  campaignName: string,
  signal?: AbortSignal,
): Promise<CollaborationAssignmentOptionsResponse> {
  return request<CollaborationAssignmentOptionsResponse>(
    `${API_BASE}/collaboration/campaigns/${encodeURIComponent(campaignName)}/assignment-options`,
    { signal },
  );
}

export async function assignCollaborationPatients(
  requestBody: CollaborationAssignRequest,
  signal?: AbortSignal,
): Promise<CollaborationAssignResponse> {
  return request<CollaborationAssignResponse>(`${API_BASE}/collaboration/assignments`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

export async function getCollaborationExperts(
  signal?: AbortSignal,
): Promise<string[]> {
  return request<string[]>(`${API_BASE}/collaboration/experts`, { signal });
}

export interface CollaborationCompleteRequest {
  campaign_name: string;
  expert_id: string;
  patient_id: string;
  annotation_data?: Record<string, unknown>;
}

export interface CollaborationCompleteResponse {
  success: boolean;
  campaign_name: string;
  expert_id: string;
  patient_id: string;
  campaign: CollaborationCampaign;
  message: string;
}

export interface CollaborationExpertTasksResponse {
  expert_id: string;
  tasks: CrowdsourcingTask[];
}

export async function completeCollaborationTask(
  requestBody: CollaborationCompleteRequest,
  signal?: AbortSignal,
): Promise<CollaborationCompleteResponse> {
  return request<CollaborationCompleteResponse>(`${API_BASE}/collaboration/complete`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

export async function getCollaborationExpertTasks(
  expertId: string,
  remainingOnly: boolean = true,
  signal?: AbortSignal,
): Promise<CollaborationExpertTasksResponse> {
  const params = new URLSearchParams({ remaining_only: String(remainingOnly) });
  return request<CollaborationExpertTasksResponse>(
    `${API_BASE}/collaboration/experts/${encodeURIComponent(expertId)}/tasks?${params}`,
    { signal },
  );
}

// ---------------------------------------------------------------------------
// Auth  (/api/v1/auth/*)
// ---------------------------------------------------------------------------

export interface CrowdsourcingTask {
  campaign_id: string;
  patient_id: string;
  dataset_path: string;
  patient_path: string | null;
  load_path: string | null;
  modality: string | null;
}

export interface LoginResponse {
  success: boolean;
  user_id: string;
  role: 'admin' | 'expert' | string;
  expert_score: number;
  remaining_tasks: CrowdsourcingTask[];
  assigned_tasks: CrowdsourcingTask[];
  message: string;
}

export async function login(
  username: string,
  password: string,
  signal?: AbortSignal,
): Promise<LoginResponse> {
  return request<LoginResponse>(`${API_BASE}/auth/login`, {
    method: 'POST',
    signal,
    body: JSON.stringify({ username, password }),
  });
}

// ---------------------------------------------------------------------------
// Annotations  (/api/v1/annotations/*)
// ---------------------------------------------------------------------------

export interface AnnotationStoreRequest {
  user_id: string;
  study_path: string;
  annotation_type: string;
  slice_annotations: Array<Record<string, unknown>>;
  study_metadata?: Record<string, unknown>;
  vlm_labels?: Record<string, unknown>;
}

export interface VlmLabelDecisionRequest {
  user_id: string;
  study_path: string;
  slice_idx: number;
  view_type: string;
  label: string;
  action: 'accepted' | 'rejected';
  suggested_labels?: string[];
  source?: string;
}

export interface VlmLabelDecisionResponse {
  success: boolean;
  user_id: string;
  study_path: string;
  slice_idx: number;
  view_type: string;
  label: string;
  action: 'accepted' | 'rejected';
  saved_labels: string[];
  message: string;
}

export interface AnnotationStoreResponse {
  success: boolean;
  user_id: string;
  study_path: string;
  annotation_count: number;
  message: string;
}

export interface AnnotationLoadResponse {
  found: boolean;
  user_id: string;
  study_path: string;
  annotation_count: number;
  data: Record<string, unknown> | null;
  message: string;
}

export interface AnnotationExportResponse {
  success: boolean;
  user_id: string;
  study_path: string;
  output_format: string;
  export_path: string | null;
  message: string;
}

export async function storeAnnotations(
  requestBody: AnnotationStoreRequest,
  signal?: AbortSignal,
): Promise<AnnotationStoreResponse> {
  return request<AnnotationStoreResponse>(`${API_BASE}/annotations/store`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

export async function reviewVlmLabelSuggestion(
  requestBody: VlmLabelDecisionRequest,
  signal?: AbortSignal,
): Promise<VlmLabelDecisionResponse> {
  return request<VlmLabelDecisionResponse>(`${API_BASE}/annotations/vlm-label-decision`, {
    method: 'POST',
    signal,
    body: JSON.stringify(requestBody),
  });
}

export async function loadAnnotations(
  userId: string,
  studyPath: string,
  signal?: AbortSignal,
): Promise<AnnotationLoadResponse> {
  return request<AnnotationLoadResponse>(`${API_BASE}/annotations/load`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      user_id: userId,
      study_path: studyPath,
    }),
  });
}

export async function exportAnnotations(
  userId: string,
  studyPath: string,
  outputFormat: string = 'json',
  signal?: AbortSignal,
): Promise<AnnotationExportResponse> {
  return request<AnnotationExportResponse>(`${API_BASE}/annotations/export`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      user_id: userId,
      study_path: studyPath,
      output_format: outputFormat,
    }),
  });
}
