"""
SegPro-Med Behavioral Category Calculator

This module computes behavioral categories from session counters
and event data. It implements the 6 key categories:

1. AI Dependency Level - How much users rely on AI tools
2. Speed Profile - How fast users annotate
3. Experience Level - Proficiency patterns
4. Modality Expertise - Specialization in imaging types
5. VLM Usage Pattern - Vision-Language Model interaction
6. Crowdsourcing Participation - Engagement in collaborative annotation
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Category Enums
# ============================================================================

class AIDependencyLevel(Enum):
    """AI Dependency Level categories"""
    AI_HEAVY = "ai_heavy"           # AI Assistance Rate > 60%
    HYBRID = "hybrid"               # AI Assistance Rate 30-60%
    MANUAL_FIRST = "manual_first"   # AI Assistance Rate < 30%
    UNKNOWN = "unknown"


class SpeedProfile(Enum):
    """Speed Profile categories"""
    SPEED_DEMON = "speed_demon"           # < 45 sec/annotation avg
    BALANCED = "balanced"                 # 45-120 sec/annotation
    METICULOUS = "meticulous_reviewer"    # > 120 sec/annotation
    UNKNOWN = "unknown"


class ExperienceLevel(Enum):
    """Experience Level categories"""
    NOVICE = "novice"             # High variation, frequent tool changes, high edit rate
    INTERMEDIATE = "intermediate"  # Moderate patterns
    EXPERT = "expert"             # Consistent, minimal switching, low edit rate
    UNKNOWN = "unknown"


class ModalityExpertise(Enum):
    """Modality Expertise categories"""
    BRAIN_MRI_SPECIALIST = "brain_mri_specialist"
    MAMMOGRAPHY_EXPERT = "mammography_expert"
    ABDOMINAL_SPECIALIST = "abdominal_specialist"
    GENERALIST = "generalist"
    UNKNOWN = "unknown"


class VLMUsagePattern(Enum):
    """VLM Usage Pattern categories"""
    VLM_POWER_USER = "vlm_power_user"       # Regular VLM use (>20% sessions)
    VLM_OCCASIONAL = "vlm_occasional_user"   # Sporadic VLM use (5-20%)
    VLM_NON_USER = "vlm_non_user"           # No VLM engagement
    UNKNOWN = "unknown"


class CrowdsourcingParticipation(Enum):
    """Crowdsourcing Participation categories"""
    HIGH_VOLUME = "high_volume_contributor"   # >10 assignments, >80% completion
    SELECTIVE = "selective_participant"        # 40-80% completion
    STRUGGLER = "assignment_struggler"         # <40% completion
    NON_PARTICIPANT = "non_participant"        # No participation
    UNKNOWN = "unknown"


# ============================================================================
# Profile Data Classes
# ============================================================================

@dataclass
class BehavioralProfile:
    """Complete behavioral profile for a user"""
    # Category classifications
    ai_dependency_level: str
    speed_profile: str
    experience_level: str
    modality_expertise: str
    vlm_usage_pattern: str
    crowdsourcing_participation: str
    
    # Raw metrics (for display and analysis)
    metrics: Dict[str, Any]
    
    # Confidence scores (0-1)
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": {
                "ai_dependency_level": self.ai_dependency_level,
                "speed_profile": self.speed_profile,
                "experience_level": self.experience_level,
                "modality_expertise": self.modality_expertise,
                "vlm_usage_pattern": self.vlm_usage_pattern,
                "crowdsourcing_participation": self.crowdsourcing_participation
            },
            "metrics": self.metrics,
            "confidence_scores": self.confidence_scores
        }


# ============================================================================
# Category Calculator
# ============================================================================

class BehavioralCategoryCalculator:
    """Computes behavioral categories from session data"""
    
    def __init__(self):
        """Initialize the calculator"""
        pass
    
    def compute_profile(self, counters) -> Dict[str, Any]:
        """
        Compute behavioral profile from session counters.
        
        Args:
            counters: SessionCounters object with accumulated data
            
        Returns:
            Dictionary containing categories, metrics, and confidence scores
        """
        counters_dict = counters.to_dict() if hasattr(counters, 'to_dict') else counters
        
        # Compute each category
        ai_level, ai_metrics, ai_confidence = self._compute_ai_dependency(counters_dict)
        speed, speed_metrics, speed_confidence = self._compute_speed_profile(counters_dict)
        experience, exp_metrics, exp_confidence = self._compute_experience_level(counters_dict)
        modality, mod_metrics, mod_confidence = self._compute_modality_expertise(counters_dict)
        vlm, vlm_metrics, vlm_confidence = self._compute_vlm_usage(counters_dict)
        crowd, crowd_metrics, crowd_confidence = self._compute_crowdsourcing(counters_dict)
        
        # Compute primary tool
        primary_tool_data = self._compute_primary_tool(counters_dict)
        
        # Combine all metrics
        all_metrics = {
            "ai_dependency": ai_metrics,
            "speed": speed_metrics,
            "experience": exp_metrics,
            "modality": mod_metrics,
            "vlm_usage": vlm_metrics,
            "crowdsourcing": crowd_metrics,
            "primary_tool": primary_tool_data  # Add primary tool metrics
        }
        
        # Combine confidence scores
        confidence_scores = {
            "ai_dependency_level": ai_confidence,
            "speed_profile": speed_confidence,
            "experience_level": exp_confidence,
            "modality_expertise": mod_confidence,
            "vlm_usage_pattern": vlm_confidence,
            "crowdsourcing_participation": crowd_confidence,
            "primary_tool": primary_tool_data.get("confidence", 0.0)  # Add tool confidence
        }
        
        return {
            "categories": {
                "ai_dependency_level": ai_level,
                "speed_profile": speed,
                "experience_level": experience,
                "modality_expertise": modality,
                "vlm_usage_pattern": vlm,
                "crowdsourcing_participation": crowd
            },
            "metrics": all_metrics,
            "confidence_scores": confidence_scores,
            "session_counters": counters_dict
        }
    
    def _compute_ai_dependency(self, counters: Dict) -> tuple:
        """
        Compute AI Dependency Level.
        
        Categories:
        - AI-Heavy: AI Assistance Rate > 60%
        - Hybrid: AI Assistance Rate 30-60%
        - Manual-First: AI Assistance Rate < 30%
        """
        ai_runs = counters.get('ai_segmentation_runs', 0)
        auto_runs = counters.get('automatic_segmentation_runs', 0)
        ai_assisted = counters.get('ai_assisted_annotations', 0)
        manual_annotations = counters.get('manual_annotations', 0)
        ai_accepted = counters.get('ai_suggestions_accepted', 0)
        ai_rejected = counters.get('ai_suggestions_rejected', 0)
        
        total_annotations = ai_assisted + manual_annotations
        total_ai_interactions = ai_runs + auto_runs + ai_accepted
        total_suggestions = ai_accepted + ai_rejected
        
        # Calculate AI assistance rate
        if total_annotations > 0:
            ai_assistance_rate = (ai_assisted + ai_runs + auto_runs) / (total_annotations + ai_runs + auto_runs)
        else:
            ai_assistance_rate = (ai_runs + auto_runs) / max(1, ai_runs + auto_runs + 1)
        
        # Calculate AI acceptance rate
        ai_acceptance_rate = ai_accepted / max(1, total_suggestions)
        
        # Determine category
        if total_annotations == 0 and total_ai_interactions == 0:
            category = AIDependencyLevel.UNKNOWN.value
            confidence = 0.0
        elif ai_assistance_rate > 0.6:
            category = AIDependencyLevel.AI_HEAVY.value
            confidence = min(1.0, 0.5 + ai_assistance_rate * 0.5)
        elif ai_assistance_rate >= 0.3:
            category = AIDependencyLevel.HYBRID.value
            confidence = 0.7
        else:
            category = AIDependencyLevel.MANUAL_FIRST.value
            confidence = min(1.0, 0.5 + (1 - ai_assistance_rate) * 0.5)
        
        metrics = {
            "ai_assistance_rate": round(ai_assistance_rate, 3),
            "ai_acceptance_rate": round(ai_acceptance_rate, 3),
            "ai_segmentation_runs": ai_runs,
            "automatic_segmentation_runs": auto_runs,
            "ai_assisted_annotations": ai_assisted,
            "manual_annotations": manual_annotations,
            "ai_suggestions_accepted": ai_accepted,
            "ai_suggestions_rejected": ai_rejected
        }
        
        return category, metrics, round(confidence, 3)
    
    def _compute_speed_profile(self, counters: Dict) -> tuple:
        """
        Compute Speed Profile.
        
        Categories:
        - Speed Demon: < 45 sec/annotation avg
        - Balanced: 45-120 sec/annotation
        - Meticulous Reviewer: > 120 sec/annotation
        """
        total_time_ms = counters.get('total_annotation_time_ms', 0)
        annotation_count = counters.get('annotation_count', 0)
        
        if annotation_count == 0:
            category = SpeedProfile.UNKNOWN.value
            avg_time_sec = 0
            confidence = 0.0
        else:
            avg_time_sec = (total_time_ms / 1000) / annotation_count
            
            if avg_time_sec < 45:
                category = SpeedProfile.SPEED_DEMON.value
                confidence = min(1.0, 0.6 + (45 - avg_time_sec) / 45 * 0.4)
            elif avg_time_sec <= 120:
                category = SpeedProfile.BALANCED.value
                confidence = 0.8
            else:
                category = SpeedProfile.METICULOUS.value
                confidence = min(1.0, 0.6 + min(avg_time_sec - 120, 180) / 180 * 0.4)
        
        metrics = {
            "average_annotation_time_sec": round(avg_time_sec, 2),
            "total_annotation_time_ms": total_time_ms,
            "annotation_count": annotation_count
        }
        
        return category, metrics, round(confidence, 3)
    
    def _compute_experience_level(self, counters: Dict) -> tuple:
        """
        Compute Experience Level.
        
        Categories based on:
        - Edit rate (edits / annotations)
        - Deletion rate (deletions / annotations)
        - Tool switching frequency
        """
        annotation_count = counters.get('annotation_count', 0)
        edits = counters.get('annotation_edits', 0)
        deletions = counters.get('annotation_deletions', 0)
        tool_switches = counters.get('tool_switches', 0)
        
        if annotation_count == 0:
            category = ExperienceLevel.UNKNOWN.value
            edit_rate = 0
            deletion_rate = 0
            confidence = 0.0
        else:
            edit_rate = edits / annotation_count
            deletion_rate = deletions / annotation_count
            switch_rate = tool_switches / annotation_count
            
            # Calculate experience score (lower is more expert)
            experience_score = edit_rate * 0.4 + deletion_rate * 0.3 + min(switch_rate, 1) * 0.3
            
            if edit_rate > 0.4 or deletion_rate > 0.15:
                category = ExperienceLevel.NOVICE.value
                confidence = min(1.0, 0.5 + experience_score * 0.5)
            elif edit_rate > 0.2 or deletion_rate > 0.05:
                category = ExperienceLevel.INTERMEDIATE.value
                confidence = 0.7
            else:
                category = ExperienceLevel.EXPERT.value
                confidence = min(1.0, 0.6 + (1 - experience_score) * 0.4)
        
        metrics = {
            "edit_rate": round(edit_rate, 3) if annotation_count > 0 else 0,
            "deletion_rate": round(deletion_rate, 3) if annotation_count > 0 else 0,
            "annotation_edits": edits,
            "annotation_deletions": deletions,
            "tool_switches": tool_switches,
            "annotation_count": annotation_count
        }
        
        return category, metrics, round(confidence, 3)
    
    def _compute_primary_tool(self, counters: Dict) -> Dict[str, Any]:
        """
        Compute Primary Tool preference.
        
        Returns:
            Dict with:
            - primary_tool: Most used tool (box, freehand, polygon, circle)
            - tool_distribution: Percentage breakdown
            - confidence: 0-1 (high if one tool > 60%)
        """
        tool_usage = counters.get('tool_usage', {})
        
        if not tool_usage or sum(tool_usage.values()) == 0:
            return {
                "primary_tool": "unknown",
                "tool_distribution": {},
                "confidence": 0.0
            }
        
        # Calculate total and percentages
        total_usage = sum(tool_usage.values())
        tool_percentages = {
            tool: round((count / total_usage) * 100, 1)
            for tool, count in tool_usage.items()
        }
        
        # Find primary tool (most used)
        primary_tool = max(tool_usage.items(), key=lambda x: x[1])[0]
        primary_percentage = tool_percentages[primary_tool]
        
        # Calculate confidence (high if dominant, low if mixed)
        if primary_percentage > 70:
            confidence = 0.95
        elif primary_percentage > 50:
            confidence = 0.80
        elif primary_percentage > 40:
            confidence = 0.65
        else:
            confidence = 0.50
        
        # Map tool names to friendly display names
        tool_names = {
            "box": "Rectangle",
            "freehand": "Freehand",
            "polygon": "Polygon",
            "circle": "Circle"
        }
        
        display_name = tool_names.get(primary_tool, primary_tool.capitalize())
        
        return {
            "primary_tool": primary_tool,
            "primary_tool_display": f"{display_name} ({primary_percentage}%)",
            "tool_distribution": tool_percentages,
            "confidence": round(confidence, 3),
            "total_tool_selections": total_usage
        }
    
    def _compute_modality_expertise(self, counters: Dict) -> tuple:
        """
        Compute Modality Expertise.
        
        Categories:
        - Brain MRI Specialist: >70% brain work
        - Mammography Expert: >60% MG work
        - Abdominal Specialist: >60% abdomen work
        - Generalist: 3+ dataset types
        """
        datasets = counters.get('datasets_loaded', {})
        modalities = counters.get('modalities_used', {})
        labels = counters.get('labels_created', {})
        
        total_loads = sum(datasets.values())
        
        if total_loads == 0:
            category = ModalityExpertise.UNKNOWN.value
            confidence = 0.0
            primary_dataset = None
            primary_ratio = 0
        else:
            # Find primary dataset
            primary_dataset = max(datasets.keys(), key=lambda k: datasets[k]) if datasets else None
            primary_count = datasets.get(primary_dataset, 0) if primary_dataset else 0
            primary_ratio = primary_count / total_loads
            
            # Classify based on primary dataset and ratio
            brain_keywords = ['brain', 'flair', 't1', 't1c', 't2', 'mri']
            mg_keywords = ['mg', 'mammography', 'breast']
            abdomen_keywords = ['abdomen', 'liver', 'kidney', 'spleen']
            
            is_brain = any(kw in str(primary_dataset).lower() for kw in brain_keywords)
            is_mg = any(kw in str(primary_dataset).lower() for kw in mg_keywords)
            is_abdomen = any(kw in str(primary_dataset).lower() for kw in abdomen_keywords)
            
            num_dataset_types = len(datasets)
            
            if num_dataset_types >= 3:
                category = ModalityExpertise.GENERALIST.value
                confidence = min(1.0, 0.5 + num_dataset_types * 0.1)
            elif is_brain and primary_ratio > 0.7:
                category = ModalityExpertise.BRAIN_MRI_SPECIALIST.value
                confidence = min(1.0, 0.5 + primary_ratio * 0.5)
            elif is_mg and primary_ratio > 0.6:
                category = ModalityExpertise.MAMMOGRAPHY_EXPERT.value
                confidence = min(1.0, 0.5 + primary_ratio * 0.5)
            elif is_abdomen and primary_ratio > 0.6:
                category = ModalityExpertise.ABDOMINAL_SPECIALIST.value
                confidence = min(1.0, 0.5 + primary_ratio * 0.5)
            else:
                category = ModalityExpertise.GENERALIST.value
                confidence = 0.5
        
        metrics = {
            "primary_dataset": primary_dataset,
            "primary_ratio": round(primary_ratio, 3) if total_loads > 0 else 0,
            "datasets_loaded": datasets,
            "modalities_used": modalities,
            "unique_labels": list(labels.keys()),
            "total_dataset_loads": total_loads
        }
        
        return category, metrics, round(confidence, 3)
    
    def _compute_vlm_usage(self, counters: Dict) -> tuple:
        """
        Compute VLM Usage Pattern.
        
        Categories:
        - VLM Power User: Regular use (>20% interactions involve VLM)
        - VLM Occasional User: Sporadic use (5-20%)
        - VLM Non-User: No VLM engagement
        """
        vlm_analyses = counters.get('vlm_analyses_run', 0)
        vlm_suggestions = counters.get('vlm_label_suggestions', 0)
        vlm_accepted = counters.get('vlm_labels_accepted', 0)
        voice_prompts = counters.get('voice_prompts_used', 0)
        vlm_model_usage = counters.get('vlm_model_usage', {})
        
        annotation_count = counters.get('annotation_count', 0)
        
        total_vlm_interactions = vlm_analyses + vlm_suggestions + voice_prompts
        total_interactions = annotation_count + total_vlm_interactions
        
        if total_interactions == 0:
            category = VLMUsagePattern.UNKNOWN.value
            vlm_rate = 0
            confidence = 0.0
        else:
            vlm_rate = total_vlm_interactions / total_interactions
            
            if vlm_rate > 0.2:
                category = VLMUsagePattern.VLM_POWER_USER.value
                confidence = min(1.0, 0.6 + vlm_rate * 0.4)
            elif vlm_rate >= 0.05:
                category = VLMUsagePattern.VLM_OCCASIONAL.value
                confidence = 0.7
            elif total_vlm_interactions > 0:
                category = VLMUsagePattern.VLM_OCCASIONAL.value
                confidence = 0.5
            else:
                category = VLMUsagePattern.VLM_NON_USER.value
                confidence = 0.9
        
        metrics = {
            "vlm_usage_rate": round(vlm_rate, 3) if total_interactions > 0 else 0,
            "vlm_analyses_run": vlm_analyses,
            "vlm_label_suggestions": vlm_suggestions,
            "vlm_labels_accepted": vlm_accepted,
            "voice_prompts_used": voice_prompts,
            "vlm_model_usage": vlm_model_usage,
            "total_vlm_interactions": total_vlm_interactions
        }
        
        return category, metrics, round(confidence, 3)
    
    def _compute_crowdsourcing(self, counters: Dict) -> tuple:
        """
        Compute Crowdsourcing Participation.
        
        Categories:
        - High-Volume Contributor: >10 assignments, >80% completion
        - Selective Participant: 40-80% completion
        - Assignment Struggler: <40% completion
        - Non-Participant: No crowdsourcing activity
        """
        loaded = counters.get('assignments_loaded', 0)
        submitted = counters.get('assignments_submitted', 0)
        completed = counters.get('assignments_completed', 0)
        
        if loaded == 0:
            category = CrowdsourcingParticipation.NON_PARTICIPANT.value
            completion_rate = 0
            confidence = 0.9
        else:
            completion_rate = completed / loaded
            
            if completed > 10 and completion_rate > 0.8:
                category = CrowdsourcingParticipation.HIGH_VOLUME.value
                confidence = min(1.0, 0.6 + completion_rate * 0.4)
            elif completion_rate >= 0.4:
                category = CrowdsourcingParticipation.SELECTIVE.value
                confidence = 0.7
            else:
                category = CrowdsourcingParticipation.STRUGGLER.value
                confidence = min(1.0, 0.5 + (1 - completion_rate) * 0.3)
        
        metrics = {
            "completion_rate": round(completion_rate, 3) if loaded > 0 else 0,
            "assignments_loaded": loaded,
            "assignments_submitted": submitted,
            "assignments_completed": completed
        }
        
        return category, metrics, round(confidence, 3)
    
    def _metrics_to_counters(self, metrics: Dict) -> Dict:
        """
        Convert profile metrics back to counter format for category recomputation.
        
        This allows categories to be recomputed from merged/accumulated metrics.
        """
        counters = {}
        
        # AI dependency metrics
        ai = metrics.get("ai_dependency", {})
        counters["ai_segmentation_runs"] = ai.get("ai_segmentation_runs", 0)
        counters["automatic_segmentation_runs"] = ai.get("automatic_segmentation_runs", 0)
        counters["ai_assisted_annotations"] = ai.get("ai_assisted_annotations", 0)
        counters["manual_annotations"] = ai.get("manual_annotations", 0)
        counters["ai_suggestions_accepted"] = ai.get("ai_suggestions_accepted", 0)
        counters["ai_suggestions_rejected"] = ai.get("ai_suggestions_rejected", 0)
        
        # Speed metrics
        speed = metrics.get("speed", {})
        counters["total_annotation_time_ms"] = speed.get("total_annotation_time_ms", 0)
        counters["annotation_count"] = speed.get("annotation_count", 0)
        
        # Experience metrics
        exp = metrics.get("experience", {})
        counters["annotation_edits"] = exp.get("annotation_edits", 0)
        counters["annotation_deletions"] = exp.get("annotation_deletions", 0)
        counters["tool_switches"] = exp.get("tool_switches", 0)
        # Use experience annotation_count if speed doesn't have it
        if counters["annotation_count"] == 0:
            counters["annotation_count"] = exp.get("annotation_count", 0)
        
        # Modality metrics
        mod = metrics.get("modality", {})
        counters["datasets_loaded"] = mod.get("datasets_loaded", {})
        counters["modalities_used"] = mod.get("modalities_used", {})
        counters["labels_created"] = {}  # Not stored in profile
        
        # VLM metrics
        vlm = metrics.get("vlm_usage", {})
        counters["vlm_analyses_run"] = vlm.get("vlm_analyses_run", 0)
        counters["vlm_label_suggestions"] = vlm.get("vlm_label_suggestions", 0)
        counters["vlm_labels_accepted"] = vlm.get("vlm_labels_accepted", 0)
        counters["voice_prompts_used"] = vlm.get("voice_prompts_used", 0)
        counters["vlm_model_usage"] = vlm.get("vlm_model_usage", {})
        
        # Crowdsourcing metrics
        crowd = metrics.get("crowdsourcing", {})
        counters["assignments_loaded"] = crowd.get("assignments_loaded", 0)
        counters["assignments_submitted"] = crowd.get("assignments_submitted", 0)
        counters["assignments_completed"] = crowd.get("assignments_completed", 0)
        
        # Tool usage for primary tool computation
        tool = metrics.get("primary_tool", {})
        counters["tool_usage"] = tool.get("tool_distribution", {})
        
        return counters

    def merge_profiles(self, existing: Dict, new: Dict) -> Dict:
        """
        Merge a new session profile with existing historical profile.
        
        This accumulates metrics over time for more accurate long-term
        category classification. Categories are recomputed from merged metrics.
        """
        if not existing:
            return new
        
        merged = {
            "last_updated": new.get("last_updated"),
            "session_id": new.get("session_id"),
            "user_id": new.get("user_id")
        }
        
        # Accumulate historical metrics
        existing_metrics = existing.get("metrics", {})
        new_metrics = new.get("metrics", {})
        
        merged_metrics = {}
        for category_key in new_metrics:
            if category_key not in existing_metrics:
                merged_metrics[category_key] = new_metrics[category_key]
                continue
            
            old = existing_metrics[category_key]
            new_cat = new_metrics[category_key]
            merged_cat = {}
            
            for key, value in new_cat.items():
                if key in old:
                    old_val = old[key]
                    # Accumulate counts, average rates
                    if isinstance(value, (int, float)) and isinstance(old_val, (int, float)):
                        if 'rate' in key or 'ratio' in key:
                            # Weighted average for rates
                            merged_cat[key] = round((old_val + value) / 2, 3)
                        else:
                            # Sum for counts
                            merged_cat[key] = old_val + value
                    elif isinstance(value, dict) and isinstance(old_val, dict):
                        # Merge dictionaries (e.g., vlm_model_usage)
                        merged_dict = dict(old_val)
                        for k, v in value.items():
                            merged_dict[k] = merged_dict.get(k, 0) + v
                        merged_cat[key] = merged_dict
                    elif isinstance(value, list) and isinstance(old_val, list):
                        # Combine lists and deduplicate
                        merged_cat[key] = list(set(old_val + value))
                    else:
                        merged_cat[key] = value
                else:
                    merged_cat[key] = value
            
            merged_metrics[category_key] = merged_cat
        
        merged["metrics"] = merged_metrics
        
        # Recompute categories from merged metrics to get accurate classifications
        # This ensures accumulated data leads to proper category assignment
        merged_counters = self._metrics_to_counters(merged_metrics)
        
        # Recompute each category from merged metrics - also get the recomputed metrics
        ai_level, ai_metrics, ai_conf = self._compute_ai_dependency(merged_counters)
        speed, speed_metrics, speed_conf = self._compute_speed_profile(merged_counters)
        experience, exp_metrics, exp_conf = self._compute_experience_level(merged_counters)
        modality, mod_metrics, mod_conf = self._compute_modality_expertise(merged_counters)
        vlm, vlm_metrics, vlm_conf = self._compute_vlm_usage(merged_counters)
        crowd, crowd_metrics, crowd_conf = self._compute_crowdsourcing(merged_counters)
        
        # Recompute primary tool
        primary_tool_data = self._compute_primary_tool(merged_counters)
        
        # Update metrics with correctly recomputed values (fixes rate averaging issues)
        # Keep accumulated counts from merged_metrics, update calculated rates from recomputed metrics
        for key in ["ai_assistance_rate", "ai_acceptance_rate"]:
            if key in ai_metrics:
                merged_metrics.get("ai_dependency", {})[key] = ai_metrics[key]
        for key in ["average_annotation_time_sec"]:
            if key in speed_metrics:
                merged_metrics.get("speed", {})[key] = speed_metrics[key]
        for key in ["edit_rate", "deletion_rate"]:
            if key in exp_metrics:
                merged_metrics.get("experience", {})[key] = exp_metrics[key]
        for key in ["primary_dataset", "primary_ratio"]:
            if key in mod_metrics:
                merged_metrics.get("modality", {})[key] = mod_metrics[key]
        for key in ["vlm_usage_rate"]:
            if key in vlm_metrics:
                merged_metrics.get("vlm_usage", {})[key] = vlm_metrics[key]
        for key in ["completion_rate"]:
            if key in crowd_metrics:
                merged_metrics.get("crowdsourcing", {})[key] = crowd_metrics[key]
        
        merged_metrics["primary_tool"] = primary_tool_data
        merged["metrics"] = merged_metrics
        
        merged["categories"] = {
            "ai_dependency_level": ai_level,
            "speed_profile": speed,
            "experience_level": experience,
            "modality_expertise": modality,
            "vlm_usage_pattern": vlm,
            "crowdsourcing_participation": crowd
        }
        
        merged["confidence_scores"] = {
            "ai_dependency_level": ai_conf,
            "speed_profile": speed_conf,
            "experience_level": exp_conf,
            "modality_expertise": mod_conf,
            "vlm_usage_pattern": vlm_conf,
            "crowdsourcing_participation": crowd_conf,
            "primary_tool": primary_tool_data.get("confidence", 0.0)
        }
        
        # Track session history - make a copy to avoid mutating existing
        session_history = list(existing.get("session_history", []))
        
        # Only add new session if it's different from the last one
        new_session_id = new.get("session_id")
        if not session_history or session_history[-1].get("session_id") != new_session_id:
            session_history.append({
                "session_id": new_session_id,
                "timestamp": new.get("last_updated"),
                "categories": new.get("categories", {})
            })
        
        # Keep last 50 sessions
        merged["session_history"] = session_history[-50:]
        
        # Track total sessions (count actual sessions in history)
        merged["total_sessions"] = len(merged["session_history"])
        
        return merged
