"""
SegPro-Med Analytics Module

This module provides behavioral tracking and analysis for the medical imaging
annotation platform.

Main Components:
- BehavioralTracker: Event-based tracking with enable/disable flag
- BehavioralCategoryCalculator: Computes 6 behavioral categories
- EventType: Enum of trackable event types

Usage:
    from analytics import get_tracker, set_tracker_enabled, EventType
    
    # Get the global tracker instance
    tracker = get_tracker(enabled=True)
    
    # Start a session
    tracker.start_session(user_id="john_doe")
    
    # Track events
    tracker.track_data_load(dataset="brain", modality="flair", ...)
    tracker.track_annotation_created(slice_idx=45, ...)
    tracker.track_vlm_analysis(vlm_model="MedGemma-4B", ...)
    
    # End session (computes and saves profile)
    profile = tracker.end_session()
    
    # Disable tracking if needed
    set_tracker_enabled(False)
"""

from .behavioral_tracker import (
    BehavioralTracker,
    BehavioralEvent,
    SessionCounters,
    EventType,
    get_tracker,
    set_tracker_enabled
)

from .category_calculator import (
    BehavioralCategoryCalculator,
    BehavioralProfile,
    AIDependencyLevel,
    SpeedProfile,
    ExperienceLevel,
    ModalityExpertise,
    VLMUsagePattern,
    CrowdsourcingParticipation
)

from .tracking_integration import (
    configure_tracking,
    init_tracking,
    end_tracking,
    is_tracking_active,
    get_current_user_id,
    get_user_profile,
    get_session_counters,
    track_data_load,
    track_slice_change,
    track_annotation,
    track_annotation_edit,
    track_annotation_delete,
    track_ai_segmentation,
    track_automatic_segmentation,
    track_vlm_analysis,
    track_label_suggestion,
    track_label_accepted,
    track_voice_input,
    track_tool_selection,
    track_window_adjustment,
    track_crowdsourcing_load,
    track_crowdsourcing_submit,
    track_crowdsourcing_complete,
    TrackingTimer,
    track_operation
)

__all__ = [
    # Tracker
    'BehavioralTracker',
    'BehavioralEvent', 
    'SessionCounters',
    'EventType',
    'get_tracker',
    'set_tracker_enabled',
    
    # Calculator
    'BehavioralCategoryCalculator',
    'BehavioralProfile',
    
    # Category Enums
    'AIDependencyLevel',
    'SpeedProfile',
    'ExperienceLevel',
    'ModalityExpertise',
    'VLMUsagePattern',
    'CrowdsourcingParticipation'    
    # Integration helpers
    'configure_tracking',
    'init_tracking',
    'end_tracking',
    'is_tracking_active',
    'get_current_user_id',
    'get_user_profile',
    'get_session_counters',
    'track_data_load',
    'track_slice_change',
    'track_annotation',
    'track_annotation_edit',
    'track_annotation_delete',
    'track_ai_segmentation',
    'track_automatic_segmentation',
    'track_vlm_analysis',
    'track_label_suggestion',
    'track_label_accepted',
    'track_voice_input',
    'track_tool_selection',
    'track_window_adjustment',
    'track_crowdsourcing_load',
    'track_crowdsourcing_submit',
    'track_crowdsourcing_complete',
    'TrackingTimer',
    'track_operation',]
