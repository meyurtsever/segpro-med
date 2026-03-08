"""
Experts Tab for Admin Users to view behavioral analytics profiles and sessions.

This tab allows admins to:
- View all expert profiles in a table format with pagination
- See detailed metrics for AI usage, speed, experience, modality expertise
- Explore individual session histories

UI Design: Clean, minimal dark theme with optimized spacing
"""

import gradio as gr
import logging
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

BEHAVIORAL_ANALYTICS_PATH = "db/behavioral_analytics"


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_all_users() -> List[str]:
    """Get list of all users with behavioral analytics data."""
    users = []
    if os.path.exists(BEHAVIORAL_ANALYTICS_PATH):
        for user_dir in os.listdir(BEHAVIORAL_ANALYTICS_PATH):
            user_path = os.path.join(BEHAVIORAL_ANALYTICS_PATH, user_dir)
            if os.path.isdir(user_path):
                profile_path = os.path.join(user_path, "profile.json")
                if os.path.exists(profile_path):
                    users.append(user_dir)
    return sorted(users)


def load_user_profile(user_id: str) -> Optional[Dict]:
    """Load the profile.json for a specific user."""
    profile_path = os.path.join(BEHAVIORAL_ANALYTICS_PATH, user_id, "profile.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading profile for {user_id}: {e}")
    return None


def get_user_sessions(user_id: str) -> List[Dict]:
    """Get list of session files for a user with basic metadata."""
    sessions = []
    sessions_path = os.path.join(BEHAVIORAL_ANALYTICS_PATH, user_id, "sessions")
    
    if not os.path.exists(sessions_path):
        return sessions
    
    for session_file in os.listdir(sessions_path):
        if session_file.endswith('.jsonl'):
            session_id = session_file.replace('.jsonl', '')
            session_filepath = os.path.join(sessions_path, session_file)
            
            try:
                stat = os.stat(session_filepath)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                
                event_count = 0
                first_event = None
                last_event = None
                
                with open(session_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            event_count += 1
                            try:
                                event = json.loads(line)
                                if first_event is None:
                                    first_event = event
                                last_event = event
                            except:
                                pass
                
                start_time = first_event.get('timestamp') if first_event else None
                end_time = last_event.get('timestamp') if last_event else None
                
                sessions.append({
                    'session_id': session_id,
                    'file_path': session_filepath,
                    'modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_count': event_count,
                    'start_time': start_time,
                    'end_time': end_time
                })
            except Exception as e:
                logger.error(f"Error reading session {session_file}: {e}")
    
    sessions.sort(key=lambda x: x['modified'], reverse=True)
    return sessions


def load_session_events(user_id: str, session_id: str) -> List[Dict]:
    """Load all events from a specific session file."""
    session_path = os.path.join(BEHAVIORAL_ANALYTICS_PATH, user_id, "sessions", f"{session_id}.jsonl")
    events = []
    
    if os.path.exists(session_path):
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
    
    return events


# =============================================================================
# Data Formatting Functions
# =============================================================================

def format_metric_value(value: Any, metric_type: str = 'number') -> str:
    """Format a metric value for display."""
    if value is None:
        return "—"
    
    if metric_type == 'percentage':
        if isinstance(value, (int, float)):
            return f"{value * 100:.1f}%"
    elif metric_type == 'time_sec':
        if isinstance(value, (int, float)):
            return f"{value:.2f}s"
    elif metric_type == 'number':
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    
    return str(value)


def get_users_dataframe() -> pd.DataFrame:
    """Create a DataFrame with all user profiles for the table."""
    users = get_all_users()
    
    if not users:
        return pd.DataFrame(columns=['User', 'Experience', 'AI Style', 'Speed', 'Annotations', 'Sessions', 'Last Active'])
    
    rows = []
    for user_id in users:
        profile = load_user_profile(user_id)
        if not profile:
            continue
        
        categories = profile.get('categories', {})
        metrics = profile.get('metrics', {})
        last_updated = profile.get('last_updated', '')
        
        annotation_count = metrics.get('speed', {}).get('annotation_count', 0)
        session_count = len(get_user_sessions(user_id))
        
        # Format last updated
        last_active = ""
        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                last_active = dt.strftime('%m/%d %H:%M')
            except:
                pass
        
        # Helper to format category values - convert 'unknown' to dash
        def format_category(value):
            if not value or value == 'unknown':
                return '—'
            return value.replace('_', ' ').title()
        
        rows.append({
            'User': user_id,
            'Experience': format_category(categories.get('experience_level')),
            'AI Style': format_category(categories.get('ai_dependency_level')),
            'Speed': format_category(categories.get('speed_profile')),
            'Annotations': annotation_count,
            'Sessions': session_count,
            'Last Active': last_active
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# HTML Component Builders - Compact Dark Theme
# =============================================================================

def create_category_badge(value: str) -> str:
    """Create a compact badge for category values."""
    badge_config = {
        'unknown': ('#4b5563', '—'),  # Gray for unknown/no data
        'manual_first': ('#22c55e', 'Manual First'),
        'hybrid': ('#eab308', 'Hybrid'),
        'ai_reliant': ('#3b82f6', 'AI Reliant'),
        'ai_heavy': ('#3b82f6', 'AI Heavy'),  # Added alias
        'speed_demon': ('#ef4444', 'Speed Demon'),
        'balanced': ('#eab308', 'Balanced'),
        'methodical': ('#22c55e', 'Methodical'),
        'expert': ('#22c55e', 'Expert'),
        'intermediate': ('#eab308', 'Intermediate'),
        'novice': ('#f97316', 'Novice'),
        'brain_mri_specialist': ('#a855f7', 'Brain MRI'),
        'abdomen_ct_specialist': ('#06b6d4', 'Abdomen CT'),
        'mammography_expert': ('#ec4899', 'Mammography'),  # Added
        'abdominal_specialist': ('#06b6d4', 'Abdominal'),  # Added
        'multi_modality_expert': ('#ec4899', 'Multi-Modal'),
        'generalist': ('#71717a', 'Generalist'),
        'vlm_power_user': ('#3b82f6', 'Power User'),
        'vlm_occasional_user': ('#eab308', 'Occasional'),
        'vlm_non_user': ('#71717a', 'Non-User'),
        'active_contributor': ('#22c55e', 'Active'),
        'selective_participant': ('#eab308', 'Selective'),
        'non_participant': ('#71717a', 'None'),
    }
    
    color, label = badge_config.get(value, ('#71717a', value.replace('_', ' ').title()))
    
    return f'<span style="background:{color}18;border:1px solid {color}40;color:{color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500;">{label}</span>'


def create_metric_row(label: str, value: str) -> str:
    """Create a single metric row."""
    return f'''<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #252525;">
        <span style="color:#9ca3af;font-size:12px;">{label}</span>
        <span style="color:#f5f5f5;font-weight:600;font-size:12px;">{value}</span>
    </div>'''


def create_profile_display_html(profile: Dict, user_id: str) -> str:
    """Create the profile display HTML - compact version."""
    if not profile:
        return '''
        <div style="text-align:center;padding:48px 20px;background:#141414;border:1px solid #252525;border-radius:10px;">
            <div style="color:#6b7280;font-size:14px;">Select a user from the table above to view their profile</div>
        </div>
        '''
    
    categories = profile.get('categories', {})
    metrics = profile.get('metrics', {})
    confidence_scores = profile.get('confidence_scores', {})
    last_updated = profile.get('last_updated', '—')
    
    if last_updated and last_updated != '—':
        try:
            dt = datetime.fromisoformat(last_updated)
            last_updated = dt.strftime('%Y-%m-%d %H:%M')
        except:
            pass
    
    # Get metrics
    ai_metrics = metrics.get('ai_dependency', {})
    speed_metrics = metrics.get('speed', {})
    experience_metrics = metrics.get('experience', {})
    modality_metrics = metrics.get('modality', {})
    vlm_metrics = metrics.get('vlm_usage', {})
    crowdsourcing_metrics = metrics.get('crowdsourcing', {})
    primary_tool = metrics.get('primary_tool', {})
    
    # Build category badges row
    category_config = [
        ('ai_dependency_level', 'AI'),
        ('speed_profile', 'Speed'),
        ('experience_level', 'Exp'),
        ('modality_expertise', 'Modality'),
        ('vlm_usage_pattern', 'VLM'),
        ('crowdsourcing_participation', 'Contrib'),
    ]
    
    category_badges = ""
    for cat_key, cat_label in category_config:
        value = categories.get(cat_key, 'unknown')
        conf = confidence_scores.get(cat_key, 0)
        badge = create_category_badge(value)
        category_badges += f'''<div style="display:flex;flex-direction:column;align-items:center;gap:2px;">
            <span style="color:#6b7280;font-size:9px;text-transform:uppercase;">{cat_label}</span>
            {badge}
            <span style="color:#4b5563;font-size:9px;">{conf*100:.0f}%</span>
        </div>'''
    
    # Build compact metric cards
    # Handle both old and new primary_tool formats, and handle 'unknown' values
    if isinstance(primary_tool, dict):
        raw_tool = primary_tool.get('primary_tool', 'unknown')
        if raw_tool == 'unknown' or not raw_tool:
            primary_tool_display = '—'
        else:
            # Use display name if available, otherwise format the raw tool name
            primary_tool_display = primary_tool.get('primary_tool_display', raw_tool.replace('_', ' ').title())
    else:
        primary_tool_display = '—'
    
    ai_card = f'''<div style="background:#1a1a1a;border:1px solid #252525;border-radius:8px;padding:12px;flex:1;">
        <div style="color:#60a5fa;font-size:12px;font-weight:600;margin-bottom:8px;border-bottom:1px solid #3b82f630;padding-bottom:6px;">AI Usage</div>
        {create_metric_row("Assistance Rate", format_metric_value(ai_metrics.get('ai_assistance_rate'), 'percentage'))}
        {create_metric_row("Acceptance Rate", format_metric_value(ai_metrics.get('ai_acceptance_rate'), 'percentage'))}
        {create_metric_row("Manual", str(ai_metrics.get('manual_annotations', 0)))}
        {create_metric_row("AI Runs", str(ai_metrics.get('ai_segmentation_runs', 0)))}
    </div>'''
    
    speed_card = f'''<div style="background:#1a1a1a;border:1px solid #252525;border-radius:8px;padding:12px;flex:1;">
        <div style="color:#4ade80;font-size:12px;font-weight:600;margin-bottom:8px;border-bottom:1px solid #22c55e30;padding-bottom:6px;">Speed & Quality</div>
        {create_metric_row("Avg Time", format_metric_value(speed_metrics.get('average_annotation_time_sec'), 'time_sec'))}
        {create_metric_row("Total", str(speed_metrics.get('annotation_count', 0)))}
        {create_metric_row("Edit Rate", format_metric_value(experience_metrics.get('edit_rate'), 'percentage'))}
        {create_metric_row("Delete Rate", format_metric_value(experience_metrics.get('deletion_rate'), 'percentage'))}
    </div>'''
    
    # Safely get primary_dataset - handle None values
    primary_dataset = modality_metrics.get('primary_dataset')
    primary_dataset_display = primary_dataset.title() if primary_dataset else '—'
    
    work_card = f'''<div style="background:#1a1a1a;border:1px solid #252525;border-radius:8px;padding:12px;flex:1;">
        <div style="color:#c084fc;font-size:12px;font-weight:600;margin-bottom:8px;border-bottom:1px solid #a855f730;padding-bottom:6px;">Work Style</div>
        {create_metric_row("Tool", primary_tool_display)}
        {create_metric_row("Dataset", primary_dataset_display)}
        {create_metric_row("VLM Rate", format_metric_value(vlm_metrics.get('vlm_usage_rate'), 'percentage'))}
        {create_metric_row("VLM Runs", str(vlm_metrics.get('vlm_analyses_run', 0)))}
    </div>'''
    
    crowd_card = f'''<div style="background:#1a1a1a;border:1px solid #252525;border-radius:8px;padding:12px;flex:1;">
        <div style="color:#fb923c;font-size:12px;font-weight:600;margin-bottom:8px;border-bottom:1px solid #f9731630;padding-bottom:6px;">Crowdsourcing</div>
        {create_metric_row("Loaded", str(crowdsourcing_metrics.get('assignments_loaded', 0)))}
        {create_metric_row("Completed", str(crowdsourcing_metrics.get('assignments_completed', 0)))}
        {create_metric_row("Submitted", str(crowdsourcing_metrics.get('assignments_submitted', 0)))}
    </div>'''
    
    # Experience color for header
    experience = categories.get('experience_level', 'unknown')
    exp_colors = {'expert': '#22c55e', 'intermediate': '#eab308', 'novice': '#f97316'}
    hdr_color = exp_colors.get(experience, '#3b82f6')
    
    return f'''
    <div style="background:#141414;border:1px solid #252525;border-radius:10px;padding:16px;">
        <!-- Header -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #252525;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="width:44px;height:44px;background:{hdr_color}15;border:1px solid {hdr_color}40;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;">👤</div>
                <div>
                    <div style="color:#f5f5f5;font-weight:700;font-size:18px;">{user_id}</div>
                    <div style="color:#6b7280;font-size:11px;">Last active: {last_updated}</div>
                </div>
            </div>
            <div style="background:#22c55e15;border:1px solid #22c55e30;padding:4px 12px;border-radius:6px;">
                <span style="color:#22c55e;font-size:11px;font-weight:500;">● Active</span>
            </div>
        </div>
        
        <!-- Categories -->
        <div style="background:#0f0f0f;border:1px solid #252525;border-radius:8px;padding:12px;margin-bottom:12px;">
            <div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;">
                {category_badges}
            </div>
        </div>
        
        <!-- Metrics Grid -->
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            {ai_card}
            {speed_card}
            {work_card}
            {crowd_card}
        </div>
    </div>
    '''


def create_sessions_html(user_id: str) -> str:
    """Create the sessions list HTML - improved compact version."""
    if not user_id:
        return '''
        <div style="text-align:center;padding:32px 20px;background:#141414;border:1px solid #252525;border-radius:10px;">
            <div style="color:#6b7280;font-size:13px;">Session history will appear here after selecting a user</div>
        </div>
        '''
    
    sessions = get_user_sessions(user_id)
    
    if not sessions:
        return '''
        <div style="text-align:center;padding:32px 20px;background:#141414;border:1px solid #252525;border-radius:10px;">
            <div style="color:#6b7280;font-size:13px;">No sessions found for this user</div>
        </div>
        '''
    
    sessions_html = ""
    for i, session in enumerate(sessions):
        # Calculate duration
        duration_str = "—"
        if session.get('start_time') and session.get('end_time'):
            try:
                start_dt = datetime.fromisoformat(session['start_time'])
                end_dt = datetime.fromisoformat(session['end_time'])
                duration = end_dt - start_dt
                mins = duration.total_seconds() / 60
                if mins < 1:
                    duration_str = f"{duration.total_seconds():.0f}s"
                else:
                    duration_str = f"{mins:.1f}m"
            except:
                pass
        
        # Format date
        date_display = "—"
        if session.get('start_time'):
            try:
                dt = datetime.fromisoformat(session['start_time'])
                date_display = dt.strftime('%m/%d %H:%M')
            except:
                date_display = session['modified'][:10]
        
        # Load events for breakdown
        events = load_session_events(user_id, session['session_id'])
        event_types = {}
        for event in events:
            evt_type = event.get('event_type', 'unknown')
            event_types[evt_type] = event_types.get(evt_type, 0) + 1
        
        # Build event summary - compact
        event_summary = []
        type_colors = {
            'annotation_created': '#3b82f6',
            'annotation_edited': '#eab308',
            'annotation_deleted': '#ef4444',
            'data_load': '#a855f7',
            'tool_usage': '#06b6d4',
        }
        
        for evt_type in ['annotation_created', 'annotation_edited', 'annotation_deleted', 'data_load', 'tool_usage']:
            if evt_type in event_types:
                color = type_colors.get(evt_type, '#71717a')
                short = evt_type.replace('annotation_', '').replace('_', ' ')[:6]
                event_summary.append(f'<span style="color:{color};font-size:10px;">{short}:{event_types[evt_type]}</span>')
        
        event_text = " · ".join(event_summary[:4]) if event_summary else '<span style="color:#4b5563;font-size:10px;">No events</span>'
        
        sessions_html += f'''
        <div style="display:flex;align-items:center;gap:12px;padding:10px 12px;background:#1a1a1a;border:1px solid #252525;border-radius:6px;margin-bottom:6px;">
            <div style="min-width:60px;color:#f5f5f5;font-size:12px;font-weight:500;">{date_display}</div>
            <div style="width:1px;height:24px;background:#252525;"></div>
            <div style="min-width:40px;color:#9ca3af;font-size:11px;">{duration_str}</div>
            <div style="width:1px;height:24px;background:#252525;"></div>
            <div style="min-width:36px;color:#60a5fa;font-size:11px;font-weight:500;">{session['event_count']} evt</div>
            <div style="width:1px;height:24px;background:#252525;"></div>
            <div style="flex:1;overflow:hidden;">{event_text}</div>
            <div style="color:#4b5563;font-size:9px;font-family:monospace;">#{i+1}</div>
        </div>
        '''
    
    return f'''
    <div style="background:#141414;border:1px solid #252525;border-radius:10px;padding:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="color:#f5f5f5;font-size:14px;font-weight:600;">Session History</div>
            <span style="background:#a855f720;color:#c084fc;padding:2px 10px;border-radius:9999px;font-size:11px;font-weight:500;">{len(sessions)}</span>
        </div>
        <div style="max-height:280px;overflow-y:auto;">
            {sessions_html}
        </div>
    </div>
    '''


# =============================================================================
# Main Tab Creation
# =============================================================================

def create_experts_tab():
    """Create the Experts tab for viewing behavioral analytics (admin only)."""
    
    def refresh_table():
        """Refresh the users table."""
        return get_users_dataframe()
    
    def on_table_select(evt: gr.SelectData, df: pd.DataFrame):
        """Handle table row selection."""
        if evt.index is None or len(evt.index) < 1:
            return create_profile_display_html(None, ""), create_sessions_html(None)
        
        row_idx = evt.index[0]
        if row_idx >= len(df):
            return create_profile_display_html(None, ""), create_sessions_html(None)
        
        user_id = df.iloc[row_idx]['User']
        profile = load_user_profile(user_id)
        
        return create_profile_display_html(profile, user_id), create_sessions_html(user_id)
    
    # Tab definition
    with gr.Tab("Expert Profiles", visible=False, id=6) as experts_tab:
        
        # Row 1: Header
        gr.HTML('''
        <div style="background:#141414;border:1px solid #252525;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="color:#f5f5f5;font-size:20px;font-weight:700;">Expert Behavioral Analytics</div>
                    <div style="color:#6b7280;font-size:12px;margin-top:2px;">Monitor annotation patterns, session histories, and performance metrics</div>
                </div>
                <div style="background:#3b82f615;border:1px solid #3b82f630;padding:6px 14px;border-radius:6px;">
                    <span style="color:#60a5fa;font-size:12px;font-weight:500;">Admin Only</span>
                </div>
            </div>
        </div>
        ''')
        
        # Row 2: Users Table
        with gr.Row():
            refresh_btn = gr.Button("Refresh", size="sm", scale=0, min_width=80)
        
        users_table = gr.Dataframe(
            value=get_users_dataframe(),
            headers=['User', 'Experience', 'AI Style', 'Speed', 'Annotations', 'Sessions', 'Last Active'],
            datatype=['str', 'str', 'str', 'str', 'number', 'number', 'str'],
            interactive=False,
            wrap=True,
            max_height=200
        )
        
        # Row 3: Details (Profile + Sessions)
        gr.HTML('<div style="height:12px;"></div>')
        
        profile_html = gr.HTML(value=create_profile_display_html(None, ""))
        
        gr.HTML('<div style="height:8px;"></div>')
        
        sessions_html = gr.HTML(value=create_sessions_html(None))
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_table,
            outputs=[users_table]
        )
        
        users_table.select(
            fn=on_table_select,
            inputs=[users_table],
            outputs=[profile_html, sessions_html]
        )
    
    return {
        'tab': experts_tab,
        'users_table': users_table,
        'profile_html': profile_html,
        'sessions_html': sessions_html,
    }
