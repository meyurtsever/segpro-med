"""
SegPro-Med Behavioral Tracking Integration

This module provides integration between the behavioral tracker and the application.
It wraps existing handler methods to add tracking without modifying the original code.

Usage:
    from analytics.tracking_integration import init_tracking, get_tracker
    
    # Initialize tracking for a user session
    init_tracking(user_id="john_doe")
    
    # Get tracker for manual event tracking
    tracker = get_tracker()
    tracker.track_annotation_created(...)
"""

import logging
import time
import functools
from typing import Any, Callable, Optional, Dict

from .behavioral_tracker import BehavioralTracker, EventType, get_tracker, set_tracker_enabled

logger = logging.getLogger(__name__)


# ============================================================================
# Tracking Configuration
# ============================================================================

class TrackingConfig:
    """Configuration for behavioral tracking"""
    
    # Enable/disable tracking globally
    enabled: bool = True
    
    # Storage location
    base_dir: str = "db/behavioral_analytics"
    
    # Session timeout (auto-end session if no events for this duration)
    session_timeout_minutes: int = 30
    
    # Events to skip tracking (for performance)
    skip_events: set = set()


_config = TrackingConfig()


def configure_tracking(
    enabled: bool = True,
    base_dir: str = "db/behavioral_analytics",
    session_timeout_minutes: int = 30
):
    """
    Configure behavioral tracking settings.
    
    Args:
        enabled: Whether tracking is enabled
        base_dir: Base directory for storage
        session_timeout_minutes: Session timeout in minutes
    """
    _config.enabled = enabled
    _config.base_dir = base_dir
    _config.session_timeout_minutes = session_timeout_minutes
    
    # Update global tracker
    set_tracker_enabled(enabled)
    
    logger.info(f"Behavioral tracking configured: enabled={enabled}, base_dir={base_dir}")


# ============================================================================
# Session Management
# ============================================================================

_current_user_id: Optional[str] = None
_session_start_time: Optional[float] = None


def init_tracking(user_id: str) -> str:
    """
    Initialize tracking for a user session.
    
    Call this when a user logs in or starts using the application.
    
    Args:
        user_id: The user ID from authentication
        
    Returns:
        Session ID
    """
    global _current_user_id, _session_start_time
    
    if not _config.enabled:
        logger.debug("Tracking disabled, skipping session initialization")
        return ""
    
    tracker = get_tracker(base_dir=_config.base_dir, enabled=_config.enabled)
    
    # End any existing session for this user
    if _current_user_id == user_id and tracker.is_session_active:
        tracker.end_session()
    
    _current_user_id = user_id
    _session_start_time = time.time()
    
    session_id = tracker.start_session(user_id)
    logger.info(f"Behavioral tracking session started for user: {user_id}, session: {session_id}")
    
    return session_id


def end_tracking() -> Optional[Dict[str, Any]]:
    """
    End the current tracking session.
    
    Call this when a user logs out or closes the application.
    
    Returns:
        The computed behavioral profile or None
    """
    global _current_user_id, _session_start_time
    
    if not _config.enabled:
        return None
    
    tracker = get_tracker()
    profile = tracker.end_session()
    
    _current_user_id = None
    _session_start_time = None
    
    if profile:
        logger.info(f"Behavioral tracking session ended, profile computed")
    
    return profile


def get_current_user_id() -> Optional[str]:
    """Get the current tracked user ID"""
    return _current_user_id


def is_tracking_active() -> bool:
    """Check if tracking is currently active"""
    tracker = get_tracker()
    return _config.enabled and tracker.is_session_active


# ============================================================================
# Event Tracking Helpers
# ============================================================================

def track_data_load(
    dataset: str,
    modality: str,
    patient_id: str = None,
    total_slices: int = None,
    directory: str = None
):
    """Track when data is loaded"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    
    # Infer dataset type from directory or patient_id
    dataset_type = _infer_dataset_type(directory or dataset)
    
    tracker.track_data_load(
        dataset=dataset_type,
        modality=modality,
        patient_id=patient_id,
        total_slices=total_slices,
        view_orientation="axial"
    )


def track_slice_change(from_slice: int, to_slice: int, method: str = "slider"):
    """Track slice navigation"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_slice_navigation(
        from_slice=from_slice,
        to_slice=to_slice,
        navigation_type=method
    )


def track_annotation(
    slice_idx: int,
    annotation_type: str,
    label: str,
    ai_assisted: bool = False,
    duration_ms: int = None,
    bbox: Dict = None
):
    """Track annotation creation and update last annotation timestamp for filtering automatic tool switches"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_annotation_created(
        slice_idx=slice_idx,
        annotation_type=annotation_type,
        label=label,
        ai_assisted=ai_assisted,
        duration_ms=duration_ms,
        bbox=bbox
    )
    
    # Update last annotation timestamp in app state to filter automatic tool switches
    # This prevents tracking "polygon" when tool auto-switches to "pan" after annotation
    try:
        import time
        import app
        if hasattr(app, 'app_instance') and app.app_instance and hasattr(app.app_instance, 'state'):
            app.app_instance.state.last_annotation_timestamp = int(time.time() * 1000)
    except Exception as e:
        # Silently fail if app instance not available (shouldn't happen in normal flow)
        pass


def track_annotation_edit(slice_idx: int, annotation_id: str = None):
    """Track annotation edit"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_annotation_edited(slice_idx=slice_idx, annotation_id=annotation_id)


def track_annotation_delete(slice_idx: int, annotation_id: str = None):
    """Track annotation deletion"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_annotation_deleted(slice_idx=slice_idx, annotation_id=annotation_id)


def track_tool_selection(tool_name: str, timestamp: int):
    """
    Track when user selects annotation tools.
    This is used internally by the app for duration calculation.
    
    Args:
        tool_name: Name of the tool selected (box, freehand, circle, polygon, eraser, pan)
        timestamp: Timestamp in milliseconds when tool was selected
    """
    # This function is mainly for documentation purposes
    # Tool selection is tracked in app.py state for duration calculation
    # We don't need to store it as a separate event since it's used
    # to calculate annotation_time_ms for ANNOTATION_CREATED events
    pass


def track_tool_usage(tool_type: str):
    """
    Track which annotation tools users select.
    Used to determine primary tool preference for expert profile.
    
    Args:
        tool_type: Type of tool selected (box, freehand, circle, polygon)
    """
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    
    # Track as a custom event
    tracker.track_event(
        event_type="tool_usage",
        component="image_annotator",
        action="tool_selected",
        metadata={"tool_type": tool_type}
    )
    
    # Update tool usage counter in behavioral data
    if hasattr(tracker, '_counters') and tracker._counters:
        if not hasattr(tracker._counters, 'tool_usage'):
            # Add tool_usage counter if it doesn't exist
            tracker._counters.tool_usage = {}
        
        # Increment counter for this tool type
        if tool_type not in tracker._counters.tool_usage:
            tracker._counters.tool_usage[tool_type] = 0
        tracker._counters.tool_usage[tool_type] += 1
        
        logger.debug(f"Tool usage tracked: {tool_type} (total: {tracker._counters.tool_usage[tool_type]})")


def track_ai_segmentation(
    model: str,
    processing_mode: str,
    slice_idx: int = None,
    prompt_type: str = None
):
    """Track AI segmentation run"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_ai_segmentation(
        ai_model=model,
        processing_mode=processing_mode,
        slice_idx=slice_idx,
        prompt_type=prompt_type
    )


def track_automatic_segmentation(model: str, slice_idx: int):
    """Track automatic (whole area) segmentation"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_automatic_segmentation(ai_model=model, slice_idx=slice_idx)


def track_vlm_analysis(
    model: str,
    analysis_type: str,
    slice_idx: int = None,
    has_custom_prompt: bool = False
):
    """Track VLM analysis"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_vlm_analysis(
        vlm_model=model,
        analysis_type=analysis_type,
        slice_idx=slice_idx,
        prompt="custom" if has_custom_prompt else None
    )


def track_label_suggestion(model: str, labels: list, slice_idx: int = None):
    """Track VLM label suggestion"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_vlm_label_suggestion(
        vlm_model=model,
        suggested_labels=labels,
        slice_idx=slice_idx
    )


def track_label_accepted(label: str, source: str = "vlm", slice_idx: int = None):
    """Track when a suggested label is accepted"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_label_accepted(label=label, source=source, slice_idx=slice_idx)


def track_voice_input(prompt_text: str = None, slice_idx: int = None):
    """Track voice input usage"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_voice_prompt(prompt_text=prompt_text, slice_idx=slice_idx)


def track_tool_selection(tool: str, slice_idx: int = None):
    """Track tool selection"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_tool_selected(tool_name=tool, slice_idx=slice_idx)


def track_window_adjustment(window_level: int, window_width: int, slice_idx: int = None):
    """Track window/level adjustment"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_window_adjustment(
        window_level=window_level,
        window_width=window_width,
        slice_idx=slice_idx
    )


def track_crowdsourcing_load(campaign_id: str, patient_id: str):
    """Track crowdsourcing assignment load"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_assignment_loaded(campaign_id=campaign_id, patient_id=patient_id)


def track_crowdsourcing_submit(campaign_id: str, patient_id: str, annotation_count: int = 0):
    """Track crowdsourcing submission"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_assignment_submitted(
        campaign_id=campaign_id,
        patient_id=patient_id,
        annotation_count=annotation_count
    )


def track_crowdsourcing_complete(campaign_id: str, patient_id: str):
    """Track crowdsourcing completion"""
    if not is_tracking_active():
        return
    
    tracker = get_tracker()
    tracker.track_assignment_completed(campaign_id=campaign_id, patient_id=patient_id)


# ============================================================================
# Utility Functions
# ============================================================================

def _infer_dataset_type(path: str) -> str:
    """Infer dataset type from path"""
    if not path:
        return "unknown"
    
    path_lower = path.lower()
    
    if "brain" in path_lower or "flair" in path_lower or "t1" in path_lower or "t2" in path_lower:
        return "brain"
    elif "mg" in path_lower or "mammograph" in path_lower or "breast" in path_lower:
        return "mammography"
    elif "abdomen" in path_lower or "liver" in path_lower or "kidney" in path_lower:
        return "abdomen"
    elif "cvm" in path_lower:
        return "brain"  # CVM is typically brain MRI
    else:
        return "unknown"


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get the behavioral profile for a user"""
    tracker = get_tracker()
    return tracker.get_profile(user_id)


def get_session_counters() -> Optional[Dict[str, Any]]:
    """Get current session counters for debugging/display"""
    if not is_tracking_active():
        return None
    
    tracker = get_tracker()
    return tracker.get_current_counters()


# ============================================================================
# Timing Context Manager
# ============================================================================

class TrackingTimer:
    """Context manager for tracking operation durations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = int((time.time() - self.start_time) * 1000)
        return False
    
    def get_duration_ms(self) -> int:
        return self.duration_ms or 0


# ============================================================================
# Decorator for tracking function calls
# ============================================================================

def track_operation(event_type: EventType, component: str, action: str, 
                    extract_metadata: Callable = None):
    """
    Decorator to automatically track function calls.
    
    Args:
        event_type: Type of event to track
        component: Component name
        action: Action name
        extract_metadata: Optional function to extract metadata from args/kwargs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracking_active():
                return func(*args, **kwargs)
            
            tracker = get_tracker()
            
            # Extract metadata if function provided
            metadata = {}
            if extract_metadata:
                try:
                    metadata = extract_metadata(*args, **kwargs) or {}
                except Exception as e:
                    logger.warning(f"Failed to extract metadata for tracking: {e}")
            
            # Track the event
            with TrackingTimer(func.__name__) as timer:
                result = func(*args, **kwargs)
            
            metadata['duration_ms'] = timer.get_duration_ms()
            
            tracker.track_event(
                event_type=event_type,
                component=component,
                action=action,
                metadata=metadata,
                duration_ms=timer.get_duration_ms()
            )
            
            return result
        
        return wrapper
    return decorator
