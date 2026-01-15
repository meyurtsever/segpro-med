"""
SegPro-Med Behavioral Event Tracker

This module provides event-based behavioral tracking for user interactions
in the medical imaging annotation platform. It captures events efficiently
and maintains running counters for behavioral category computation.

Storage Structure:
    db/behavioral_analytics/{user_id}/
        ├── profile.json              # Computed behavioral categories
        └── sessions/
            └── {session_id}.jsonl    # Raw events (JSON Lines format)

Tracked Categories:
    1. AI Dependency Level
    2. Speed Profile
    3. Experience Level
    4. Modality Expertise
    5. VLM Usage Pattern
    6. Crowdsourcing Participation
"""

import os
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of trackable events"""
    # Data & Session Events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    DATA_LOAD = "data_load"
    
    # Navigation Events
    SLICE_NAVIGATION = "slice_navigation"
    VIEW_CHANGE = "view_change"
    
    # Annotation Events
    ANNOTATION_CREATED = "annotation_created"
    ANNOTATION_EDITED = "annotation_edited"
    ANNOTATION_DELETED = "annotation_deleted"
    ANNOTATION_SAVED = "annotation_saved"
    
    # AI/Segmentation Events
    AI_SEGMENTATION_RUN = "ai_segmentation_run"
    AI_SEGMENTATION_COMPLETE = "ai_segmentation_complete"
    AUTOMATIC_SEGMENTATION_RUN = "automatic_segmentation_run"
    
    # VLM Events
    VLM_ANALYSIS_RUN = "vlm_analysis_run"
    VLM_ANALYSIS_COMPLETE = "vlm_analysis_complete"
    VLM_LABEL_SUGGESTION = "vlm_label_suggestion"
    VLM_LABEL_ACCEPTED = "vlm_label_accepted"
    VLM_LABEL_REJECTED = "vlm_label_rejected"
    VOICE_PROMPT_USED = "voice_prompt_used"
    
    # Label Events
    LABEL_CREATED = "label_created"
    LABEL_ASSIGNED = "label_assigned"
    SUGGESTED_LABEL_ACCEPTED = "suggested_label_accepted"
    
    # Tool Events
    TOOL_SELECTED = "tool_selected"
    WINDOW_ADJUSTMENT = "window_adjustment"
    
    # Crowdsourcing Events
    ASSIGNMENT_LOADED = "assignment_loaded"
    ASSIGNMENT_SUBMITTED = "assignment_submitted"
    ASSIGNMENT_COMPLETED = "assignment_completed"


@dataclass
class BehavioralEvent:
    """Represents a single behavioral event"""
    event_id: str
    timestamp: str
    user_id: str
    session_id: str
    event_type: str
    component: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json_line(self) -> str:
        """Convert to JSON line for append-only storage"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class SessionCounters:
    """Running counters for the current session - updated incrementally"""
    # AI Dependency Counters
    ai_segmentation_runs: int = 0
    automatic_segmentation_runs: int = 0
    manual_annotations: int = 0
    ai_assisted_annotations: int = 0
    ai_suggestions_accepted: int = 0
    ai_suggestions_rejected: int = 0
    
    # Speed Counters
    total_annotation_time_ms: int = 0
    annotation_count: int = 0
    slice_navigation_count: int = 0
    
    # Experience Counters
    annotation_edits: int = 0
    annotation_deletions: int = 0
    tool_switches: int = 0
    
    # Tool Usage Tracking (for primary tool analysis)
    tool_usage: Dict[str, int] = field(default_factory=dict)  # {tool_type: count}
    
    # Modality Counters
    datasets_loaded: Dict[str, int] = field(default_factory=dict)  # {dataset_type: count}
    modalities_used: Dict[str, int] = field(default_factory=dict)  # {modality: count}
    labels_created: Dict[str, int] = field(default_factory=dict)   # {label: count}
    
    # VLM Counters
    vlm_analyses_run: int = 0
    vlm_model_usage: Dict[str, int] = field(default_factory=dict)  # {model: count}
    vlm_label_suggestions: int = 0
    vlm_labels_accepted: int = 0
    voice_prompts_used: int = 0
    
    # Crowdsourcing Counters
    assignments_loaded: int = 0
    assignments_submitted: int = 0
    assignments_completed: int = 0
    
    # Timing
    session_start_time: Optional[float] = None
    last_annotation_start_time: Optional[float] = None
    last_tool: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class BehavioralTracker:
    """
    Main behavioral tracking class.
    
    Provides event-based tracking with:
    - Enable/disable flag for safety
    - Incremental counter updates (efficient)
    - Session-based event storage (JSON Lines)
    - Profile computation on session end
    """
    
    def __init__(self, base_dir: str = "db/behavioral_analytics", enabled: bool = True):
        """
        Initialize the behavioral tracker.
        
        Args:
            base_dir: Base directory for behavioral data storage
            enabled: Whether tracking is enabled (can be toggled at runtime)
        """
        self.base_dir = Path(base_dir)
        self._enabled = enabled
        self._current_session_id: Optional[str] = None
        self._current_user_id: Optional[str] = None
        self._counters: Optional[SessionCounters] = None
        self._session_file: Optional[Path] = None
        self._event_sequence: int = 0
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BehavioralTracker initialized (enabled={enabled}, base_dir={base_dir})")
    
    @property
    def enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable tracking"""
        self._enabled = value
        logger.info(f"BehavioralTracker {'enabled' if value else 'disabled'}")
    
    @property
    def is_session_active(self) -> bool:
        """Check if a session is currently active"""
        return self._current_session_id is not None
    
    def _get_user_dir(self, user_id: str) -> Path:
        """Get the directory for a specific user's behavioral data"""
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _get_sessions_dir(self, user_id: str) -> Path:
        """Get the sessions directory for a user"""
        sessions_dir = self._get_user_dir(user_id) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"sess_{timestamp}_{short_uuid}"
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID"""
        self._event_sequence += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"evt_{timestamp}_{self._event_sequence:04d}"
    
    def start_session(self, user_id: str) -> str:
        """
        Start a new tracking session for a user.
        
        Args:
            user_id: The user ID from authentication
            
        Returns:
            The generated session ID
        """
        if not self._enabled:
            logger.debug("Tracking disabled, session not started")
            return ""
        
        # End any existing session first
        if self.is_session_active:
            self.end_session()
        
        self._current_user_id = user_id
        self._current_session_id = self._generate_session_id()
        self._counters = SessionCounters()
        self._counters.session_start_time = time.time()
        self._event_sequence = 0
        
        # Create session file
        sessions_dir = self._get_sessions_dir(user_id)
        self._session_file = sessions_dir / f"{self._current_session_id}.jsonl"
        
        # Record session start event
        self.track_event(
            event_type=EventType.SESSION_START,
            component="system",
            action="session_started",
            metadata={"user_id": user_id}
        )
        
        logger.info(f"Started behavioral tracking session: {self._current_session_id} for user: {user_id}")
        return self._current_session_id
    
    def end_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current session and compute behavioral profile.
        
        Returns:
            The computed behavioral profile or None if no session
        """
        if not self.is_session_active:
            return None
        
        if not self._enabled:
            self._current_session_id = None
            self._current_user_id = None
            self._counters = None
            return None
        
        # Record session end event
        session_duration_ms = int((time.time() - self._counters.session_start_time) * 1000) if self._counters.session_start_time else 0
        
        self.track_event(
            event_type=EventType.SESSION_END,
            component="system",
            action="session_ended",
            metadata={
                "session_duration_ms": session_duration_ms,
                "counters_summary": self._counters.to_dict() if self._counters else {}
            }
        )
        
        # Compute and save behavioral profile
        profile = self._compute_and_save_profile()
        
        # Clean up session state
        session_id = self._current_session_id
        self._current_session_id = None
        self._current_user_id = None
        self._counters = None
        self._session_file = None
        
        logger.info(f"Ended behavioral tracking session: {session_id}")
        return profile
    
    def track_event(
        self,
        event_type: EventType,
        component: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None
    ) -> Optional[str]:
        """
        Track a behavioral event.
        
        Args:
            event_type: Type of event from EventType enum
            component: UI component that triggered the event
            action: Action performed (click, change, etc.)
            metadata: Additional event-specific data
            duration_ms: Duration of the action in milliseconds
            
        Returns:
            Event ID or None if tracking is disabled
        """
        if not self._enabled or not self.is_session_active:
            return None
        
        event = BehavioralEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            user_id=self._current_user_id,
            session_id=self._current_session_id,
            event_type=event_type.value if isinstance(event_type, EventType) else event_type,
            component=component,
            action=action,
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        
        # Write event to session file (append-only)
        self._write_event(event)
        
        # Update counters incrementally
        self._update_counters(event)
        
        logger.debug(f"Tracked event: {event.event_type} - {event.action}")
        return event.event_id
    
    def _write_event(self, event: BehavioralEvent):
        """Write event to the session file (JSON Lines format)"""
        if self._session_file:
            try:
                with open(self._session_file, 'a', encoding='utf-8') as f:
                    f.write(event.to_json_line() + '\n')
            except Exception as e:
                logger.error(f"Failed to write event: {e}")
    
    def _update_counters(self, event: BehavioralEvent):
        """Update session counters based on the event"""
        if not self._counters:
            return
        
        event_type = event.event_type
        metadata = event.metadata
        
        # AI Dependency counters
        if event_type == EventType.AI_SEGMENTATION_RUN.value:
            self._counters.ai_segmentation_runs += 1
        elif event_type == EventType.AUTOMATIC_SEGMENTATION_RUN.value:
            self._counters.automatic_segmentation_runs += 1
        elif event_type == EventType.ANNOTATION_CREATED.value:
            self._counters.annotation_count += 1
            if metadata.get('ai_assisted', False):
                self._counters.ai_assisted_annotations += 1
            else:
                self._counters.manual_annotations += 1
            # Track annotation start time and label
            self._counters.last_annotation_start_time = event.timestamp.timestamp() if hasattr(event.timestamp, 'timestamp') else time.time()
            # Track label if present
            label = metadata.get('label', '')
            if label and label != 'unlabeled':
                self._counters.labels_created[label] = self._counters.labels_created.get(label, 0) + 1
            # Track annotation duration
            if event.duration_ms:
                self._counters.total_annotation_time_ms += event.duration_ms
        
        # Speed counters
        elif event_type == EventType.SLICE_NAVIGATION.value:
            self._counters.slice_navigation_count += 1
        
        # Experience counters
        elif event_type == EventType.ANNOTATION_EDITED.value:
            self._counters.annotation_edits += 1
        elif event_type == EventType.ANNOTATION_DELETED.value:
            self._counters.annotation_deletions += 1
        elif event_type == EventType.TOOL_SELECTED.value:
            new_tool = metadata.get('tool_selected')
            if self._counters.last_tool and new_tool != self._counters.last_tool:
                self._counters.tool_switches += 1
            self._counters.last_tool = new_tool
        
        # Modality counters
        elif event_type == EventType.DATA_LOAD.value:
            dataset = metadata.get('dataset', 'unknown')
            modality = metadata.get('modality', 'unknown')
            self._counters.datasets_loaded[dataset] = self._counters.datasets_loaded.get(dataset, 0) + 1
            self._counters.modalities_used[modality] = self._counters.modalities_used.get(modality, 0) + 1
        
        elif event_type == EventType.LABEL_CREATED.value:
            label = metadata.get('label', 'unknown')
            self._counters.labels_created[label] = self._counters.labels_created.get(label, 0) + 1
        
        # VLM counters
        elif event_type == EventType.VLM_ANALYSIS_RUN.value:
            self._counters.vlm_analyses_run += 1
            model = metadata.get('vlm_model', 'unknown')
            self._counters.vlm_model_usage[model] = self._counters.vlm_model_usage.get(model, 0) + 1
        elif event_type == EventType.VLM_LABEL_SUGGESTION.value:
            self._counters.vlm_label_suggestions += 1
            # Also update vlm_model_usage for label suggestions
            model = metadata.get('vlm_model', 'unknown')
            self._counters.vlm_model_usage[model] = self._counters.vlm_model_usage.get(model, 0) + 1
        elif event_type in [EventType.VLM_LABEL_ACCEPTED.value, EventType.SUGGESTED_LABEL_ACCEPTED.value]:
            self._counters.vlm_labels_accepted += 1
            self._counters.ai_suggestions_accepted += 1
        elif event_type == EventType.VLM_LABEL_REJECTED.value:
            self._counters.ai_suggestions_rejected += 1
        elif event_type == EventType.VOICE_PROMPT_USED.value:
            self._counters.voice_prompts_used += 1
        
        # Crowdsourcing counters
        elif event_type == EventType.ASSIGNMENT_LOADED.value:
            self._counters.assignments_loaded += 1
        elif event_type == EventType.ASSIGNMENT_SUBMITTED.value:
            self._counters.assignments_submitted += 1
        elif event_type == EventType.ASSIGNMENT_COMPLETED.value:
            self._counters.assignments_completed += 1
    
    def _compute_and_save_profile(self) -> Dict[str, Any]:
        """Compute behavioral profile from counters and save to disk"""
        if not self._counters or not self._current_user_id:
            return {}
        
        # Import category calculator
        from .category_calculator import BehavioralCategoryCalculator
        
        calculator = BehavioralCategoryCalculator()
        profile = calculator.compute_profile(self._counters)
        
        # Log tool usage data for debugging
        tool_usage = self._counters.tool_usage if hasattr(self._counters, 'tool_usage') else {}
        logger.info(f"📊 Tool usage data in profile computation: {tool_usage}")
        
        # Add metadata
        profile['last_updated'] = datetime.now().isoformat()
        profile['session_id'] = self._current_session_id
        profile['user_id'] = self._current_user_id
        
        # Load existing profile and merge
        profile_path = self._get_user_dir(self._current_user_id) / "profile.json"
        existing_profile = {}
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    existing_profile = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing profile: {e}")
        
        # Merge with existing profile (accumulate historical data)
        merged_profile = calculator.merge_profiles(existing_profile, profile)
        
        # Log primary tool in final profile
        primary_tool_data = merged_profile.get('metrics', {}).get('primary_tool', {})
        logger.info(f"🎯 Primary tool in final profile: {primary_tool_data}")
        
        # Save updated profile
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(merged_profile, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved behavioral profile to: {profile_path}")
            logger.info(f"Profile categories: {merged_profile.get('categories', {})}")
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
        
        return merged_profile
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the behavioral profile for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            The user's behavioral profile or None if not found
        """
        profile_path = self._get_user_dir(user_id) / "profile.json"
        
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profile for {user_id}: {e}")
            return None
    
    def get_session_events(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all events from a specific session.
        
        Args:
            user_id: The user ID
            session_id: The session ID
            
        Returns:
            List of events from the session
        """
        session_file = self._get_sessions_dir(user_id) / f"{session_id}.jsonl"
        
        if not session_file.exists():
            return []
        
        events = []
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read session events: {e}")
        
        return events
    
    def get_current_counters(self) -> Optional[Dict[str, Any]]:
        """Get the current session counters (for debugging/display)"""
        if self._counters:
            return self._counters.to_dict()
        return None
    
    # =========================================================================
    # Convenience methods for common event types
    # =========================================================================
    
    def track_data_load(self, dataset: str, modality: str, patient_id: str = None, 
                        total_slices: int = None, view_orientation: str = "axial"):
        """Track when data is loaded"""
        self.track_event(
            event_type=EventType.DATA_LOAD,
            component="load_btn",
            action="click",
            metadata={
                "dataset": dataset,
                "modality": modality,
                "patient_id": patient_id,
                "total_slices": total_slices,
                "view_orientation": view_orientation
            }
        )
    
    def track_slice_navigation(self, from_slice: int, to_slice: int, 
                               navigation_type: str = "slider"):
        """Track slice navigation"""
        self.track_event(
            event_type=EventType.SLICE_NAVIGATION,
            component="slice_slider" if navigation_type == "slider" else "nav_btn",
            action="change",
            metadata={
                "from_slice": from_slice,
                "to_slice": to_slice,
                "navigation_type": navigation_type
            }
        )
    
    def track_annotation_created(self, slice_idx: int, annotation_type: str, 
                                 label: str, ai_assisted: bool = False,
                                 duration_ms: int = None, bbox: Dict = None):
        """Track when an annotation is created"""
        self.track_event(
            event_type=EventType.ANNOTATION_CREATED,
            component="image_annotator",
            action="annotation_created",
            metadata={
                "slice_index": slice_idx,
                "annotation_type": annotation_type,
                "label": label,
                "ai_assisted": ai_assisted,
                "bbox": bbox
            },
            duration_ms=duration_ms
        )
    
    def track_annotation_edited(self, slice_idx: int, annotation_id: str = None):
        """Track when an annotation is edited"""
        self.track_event(
            event_type=EventType.ANNOTATION_EDITED,
            component="image_annotator",
            action="annotation_edited",
            metadata={
                "slice_index": slice_idx,
                "annotation_id": annotation_id
            }
        )
    
    def track_annotation_deleted(self, slice_idx: int, annotation_id: str = None):
        """Track when an annotation is deleted"""
        self.track_event(
            event_type=EventType.ANNOTATION_DELETED,
            component="image_annotator",
            action="annotation_deleted",
            metadata={
                "slice_index": slice_idx,
                "annotation_id": annotation_id
            }
        )
    
    def track_ai_segmentation(self, ai_model: str, processing_mode: str,
                              slice_idx: int = None, prompt_type: str = None):
        """Track when AI segmentation is run"""
        self.track_event(
            event_type=EventType.AI_SEGMENTATION_RUN,
            component="annotate_btn",
            action="click",
            metadata={
                "ai_model": ai_model,
                "processing_mode": processing_mode,
                "slice_index": slice_idx,
                "prompt_type": prompt_type  # "point", "box", or "automatic"
            }
        )
    
    def track_automatic_segmentation(self, ai_model: str, slice_idx: int):
        """Track when automatic (whole area) segmentation is run"""
        self.track_event(
            event_type=EventType.AUTOMATIC_SEGMENTATION_RUN,
            component="auto_brain_annotate_btn",
            action="click",
            metadata={
                "ai_model": ai_model,
                "slice_index": slice_idx
            }
        )
    
    def track_vlm_analysis(self, vlm_model: str, analysis_type: str,
                           slice_idx: int = None, prompt: str = None):
        """Track when VLM analysis is run"""
        self.track_event(
            event_type=EventType.VLM_ANALYSIS_RUN,
            component="vlm_run_btn",
            action="click",
            metadata={
                "vlm_model": vlm_model,
                "analysis_type": analysis_type,
                "slice_index": slice_idx,
                "has_custom_prompt": prompt is not None
            }
        )
    
    def track_vlm_label_suggestion(self, vlm_model: str, suggested_labels: List[str],
                                   slice_idx: int = None):
        """Track when VLM suggests labels"""
        self.track_event(
            event_type=EventType.VLM_LABEL_SUGGESTION,
            component="suggest_labels_btn",
            action="click",
            metadata={
                "vlm_model": vlm_model,
                "suggested_labels": suggested_labels,
                "suggestion_count": len(suggested_labels),
                "slice_index": slice_idx
            }
        )
    
    def track_label_accepted(self, label: str, source: str = "vlm", slice_idx: int = None):
        """Track when a suggested label is accepted"""
        event_type = EventType.VLM_LABEL_ACCEPTED if source == "vlm" else EventType.SUGGESTED_LABEL_ACCEPTED
        self.track_event(
            event_type=event_type,
            component="accept_suggestions_btn",
            action="click",
            metadata={
                "label": label,
                "source": source,
                "slice_index": slice_idx
            }
        )
    
    def track_voice_prompt(self, prompt_text: str = None, slice_idx: int = None):
        """Track when voice prompt is used"""
        self.track_event(
            event_type=EventType.VOICE_PROMPT_USED,
            component="voice_audio_input",
            action="transcribed",
            metadata={
                "has_prompt": prompt_text is not None,
                "prompt_length": len(prompt_text) if prompt_text else 0,
                "slice_index": slice_idx
            }
        )
    
    def track_tool_selected(self, tool_name: str, slice_idx: int = None):
        """Track when a drawing tool is selected"""
        self.track_event(
            event_type=EventType.TOOL_SELECTED,
            component="tool_button",
            action="click",
            metadata={
                "tool_selected": tool_name,
                "slice_index": slice_idx
            }
        )
    
    def track_window_adjustment(self, window_level: int, window_width: int,
                                slice_idx: int = None):
        """Track when window/level is adjusted"""
        self.track_event(
            event_type=EventType.WINDOW_ADJUSTMENT,
            component="window_slider",
            action="change",
            metadata={
                "window_level": window_level,
                "window_width": window_width,
                "slice_index": slice_idx
            }
        )
    
    def track_assignment_loaded(self, campaign_id: str, patient_id: str):
        """Track when a crowdsourcing assignment is loaded"""
        self.track_event(
            event_type=EventType.ASSIGNMENT_LOADED,
            component="assignment_selector",
            action="load",
            metadata={
                "campaign_id": campaign_id,
                "patient_id": patient_id
            }
        )
    
    def track_assignment_submitted(self, campaign_id: str, patient_id: str,
                                   annotation_count: int = 0):
        """Track when a crowdsourcing assignment is submitted"""
        self.track_event(
            event_type=EventType.ASSIGNMENT_SUBMITTED,
            component="submit_annotation_btn",
            action="click",
            metadata={
                "campaign_id": campaign_id,
                "patient_id": patient_id,
                "annotation_count": annotation_count
            }
        )
    
    def track_assignment_completed(self, campaign_id: str, patient_id: str):
        """Track when a crowdsourcing assignment is marked complete"""
        self.track_event(
            event_type=EventType.ASSIGNMENT_COMPLETED,
            component="system",
            action="complete",
            metadata={
                "campaign_id": campaign_id,
                "patient_id": patient_id
            }
        )
    
    def track_label_created(self, label: str, slice_idx: int = None, 
                            is_manual: bool = True):
        """Track when a label is created/assigned"""
        self.track_event(
            event_type=EventType.LABEL_CREATED,
            component="label_input" if is_manual else "vlm_suggestion",
            action="created",
            metadata={
                "label": label,
                "slice_index": slice_idx,
                "is_manual": is_manual
            }
        )


# Global tracker instance (singleton pattern)
_tracker_instance: Optional[BehavioralTracker] = None


def get_tracker(base_dir: str = "db/behavioral_analytics", enabled: bool = True) -> BehavioralTracker:
    """
    Get or create the global behavioral tracker instance.
    
    Args:
        base_dir: Base directory for storage (only used on first call)
        enabled: Whether tracking is enabled (only used on first call)
        
    Returns:
        The global BehavioralTracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = BehavioralTracker(base_dir=base_dir, enabled=enabled)
    return _tracker_instance


def set_tracker_enabled(enabled: bool):
    """Enable or disable the global tracker"""
    tracker = get_tracker()
    tracker.enabled = enabled
