export interface MetadataRow {
  label: string;
  value: string;
}

function isFilledMetadataValue(value: unknown) {
  return value !== undefined && value !== null && value !== '';
}

function normalizeMetadataKey(key: string) {
  return key.toLowerCase().replace(/[\s_-]+/g, '');
}

function findMetadataEntry(
  metadata: Record<string, unknown> | undefined,
  keys: string[],
) {
  if (!metadata) return undefined;

  for (const key of keys) {
    const value = metadata[key];
    if (isFilledMetadataValue(value)) {
      return { key, value };
    }
  }

  const entries = Object.entries(metadata);
  for (const key of keys) {
    const normalizedKey = key.toLowerCase();
    const match = entries.find(([entryKey, value]) =>
      entryKey.toLowerCase() === normalizedKey &&
      isFilledMetadataValue(value),
    );
    if (match) return { key: match[0], value: match[1] };
  }

  return undefined;
}

export function formatMetadataValue(value: unknown): string {
  if (Array.isArray(value)) {
    if (value.some((item) => typeof item === 'object' && item !== null)) {
      try {
        return JSON.stringify(value);
      } catch {
        return String(value);
      }
    }

    return value.map((item) => formatMetadataValue(item)).join(' x ');
  }

  if (typeof value === 'object' && value !== null) {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  return String(value);
}

function formatMetadataLabel(key: string) {
  const readable = key
    .replace(/^dicom[_-]?/i, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!readable) return key;

  return readable.replace(/\b[a-z]/g, (char) => char.toUpperCase());
}

function isRenderableMetadataValue(value: unknown): boolean {
  if (!isFilledMetadataValue(value)) return false;

  if (['string', 'number', 'boolean'].includes(typeof value)) {
    return true;
  }

  try {
    const formattedValue = formatMetadataValue(value);
    return formattedValue !== '{}' &&
      formattedValue !== '[]' &&
      formattedValue.length <= 600;
  } catch {
    return false;
  }
}

export function buildMetadataRows(
  metadata: Record<string, unknown> | undefined,
  volumeShape?: number[],
): MetadataRow[] {
  const candidates: Array<[string, string[]]> = [
    ['Modality', ['Modality', 'modality']],
    ['Patient', ['PatientID', 'PatientId', 'patient_id', 'PatientName', 'patient_name']],
    ['Study', ['StudyDescription', 'study_description', 'StudyInstanceUID', 'study_id']],
    ['Series', ['SeriesDescription', 'series_description', 'SeriesNumber', 'series_number']],
    ['Study Date', ['StudyDate', 'study_date', 'AcquisitionDate', 'SeriesDate']],
    ['Manufacturer', ['Manufacturer', 'manufacturer']],
    ['Spacing', ['PixelSpacing', 'pixel_spacing', 'spacing', 'voxel_spacing', 'pixdim']],
    ['Slice Thick.', ['SliceThickness', 'slice_thickness', 'SpacingBetweenSlices']],
    ['Matrix', ['Rows', 'rows', 'Columns', 'columns']],
    ['Orientation', ['ImageOrientationPatient', 'orientation', 'canonical_view']],
  ];

  const usedKeys = new Set<string>();
  const priorityRows = candidates.flatMap(([label, keys]) => {
    if (label === 'Matrix') {
      const rowsEntry = findMetadataEntry(metadata, ['Rows', 'rows']);
      const columnsEntry = findMetadataEntry(metadata, ['Columns', 'columns']);
      if (rowsEntry && columnsEntry) {
        usedKeys.add(normalizeMetadataKey(rowsEntry.key));
        usedKeys.add(normalizeMetadataKey(columnsEntry.key));
        return [{
          label,
          value: `${formatMetadataValue(rowsEntry.value)} x ${formatMetadataValue(columnsEntry.value)}`,
        }];
      }
    }

    const entry = findMetadataEntry(metadata, keys);
    if (!entry) return [];

    usedKeys.add(normalizeMetadataKey(entry.key));
    return [{ label, value: formatMetadataValue(entry.value) }];
  });

  const rows: MetadataRow[] = [];
  const usedLabels = new Set<string>();
  const appendRow = (row: MetadataRow) => {
    const normalizedLabel = row.label.toLowerCase();
    if (!row.value || usedLabels.has(normalizedLabel)) return;

    usedLabels.add(normalizedLabel);
    rows.push(row);
  };

  if (volumeShape && volumeShape.length > 0) {
    appendRow({ label: 'Volume', value: volumeShape.join(' x ') });
  }

  priorityRows.forEach(appendRow);

  Object.entries(metadata ?? {}).forEach(([key, value]) => {
    if (usedKeys.has(normalizeMetadataKey(key)) || !isRenderableMetadataValue(value)) {
      return;
    }

    appendRow({
      label: formatMetadataLabel(key),
      value: formatMetadataValue(value),
    });
  });

  return rows.slice(0, 80);
}

export function truncateMetadataValue(value: string, maxLength = 90) {
  return value.length > maxLength
    ? `${value.slice(0, Math.max(0, maxLength - 3))}...`
    : value;
}
