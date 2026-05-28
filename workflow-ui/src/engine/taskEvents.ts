export const AI_SEGMENTATION_REVIEW_TEMPLATE_ID = 'segmentation-annotation';
export const MEDICAL_REPORT_TEMPLATE_ID = 'medgemma-slice-report';
export const VLM_LABEL_SUGGESTIONS_TEMPLATE_ID = 'vlm-label-suggestions';
export const AI_SEGMENTATION_AUTO_LOAD_EVENT = 'segpro:auto-load-ai-segmentation';
export const TASK_DATA_AUTO_LOAD_EVENT = 'segpro:auto-load-task-data';

export function supportsTaskDataAutoLoad(templateId: unknown) {
  return templateId === AI_SEGMENTATION_REVIEW_TEMPLATE_ID ||
    templateId === MEDICAL_REPORT_TEMPLATE_ID ||
    templateId === VLM_LABEL_SUGGESTIONS_TEMPLATE_ID;
}
