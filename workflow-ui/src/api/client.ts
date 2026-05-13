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
export async function loadDataFromPath(path: string): Promise<LoadDataResponse> {
  return request<LoadDataResponse>(`${API_BASE}/data/load`, {
    method: 'POST',
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
): Promise<SliceResponse> {
  const params = new URLSearchParams({
    slice: String(sliceIndex),
    view,
  });
  if (segPath) params.set('seg_path', segPath);
  return request<SliceResponse>(`${API_BASE}/data/slice/${sessionId}?${params}`);
}

/** Get volume metadata */
export async function getMetadata(sessionId: string): Promise<MetadataResponse> {
  return request<MetadataResponse>(`${API_BASE}/data/metadata/${sessionId}`);
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

/** List supported conversion types */
export async function getConversionTypes(): Promise<ConvertTypesResponse> {
  return request<ConvertTypesResponse>(`${API_BASE}/convert/types`);
}

/** Run a format conversion */
export async function runConversion(
  inputPath: string,
  conversionType: string,
  outputPath?: string,
): Promise<ConvertResponse> {
  return request<ConvertResponse>(`${API_BASE}/convert/run`, {
    method: 'POST',
    body: JSON.stringify({
      input_path: inputPath,
      conversion_type: conversionType,
      output_path: outputPath,
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

/** Browse a filesystem directory (for file picker) */
export async function browsePath(path: string): Promise<BrowseResponse> {
  return request<BrowseResponse>(
    `${API_BASE}/fs/browse?path=${encodeURIComponent(path)}`,
  );
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
// Auto Segmentation  (/api/v1/segmentation/*)
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
): Promise<AutoSegmentResponse> {
  return request<AutoSegmentResponse>(`${API_BASE}/segmentation/auto`, {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      slice_index: sliceIndex,
      view,
      config_name: configName,
    }),
  });
}

/** List available segmentation config profiles */
export async function getSegmentationConfigs(): Promise<SegmentationConfigsResponse> {
  return request<SegmentationConfigsResponse>(`${API_BASE}/segmentation/configs`);
}
