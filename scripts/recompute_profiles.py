"""
Recompute user behavioral profiles from existing session logs.

This script reads all session logs for each user and recomputes their 
behavioral profile using the updated merge logic.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.category_calculator import BehavioralCategoryCalculator


BASE_PATH = Path(r"C:\Users\Yurtsever\Downloads\segpro-med\db\behavioral_analytics")


def load_session_counters(session_file: Path) -> dict:
    """Load counters from a session file's session_end event."""
    counters = {}
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event_type") == "session_end":
                    counters = event.get("metadata", {}).get("counters_summary", {})
                    break
    except Exception as e:
        print(f"  ⚠️ Error reading {session_file.name}: {e}")
    
    return counters


def recompute_profile_for_user(user_id: str) -> dict:
    """Recompute the behavioral profile for a user from all sessions."""
    user_dir = BASE_PATH / user_id
    sessions_dir = user_dir / "sessions"
    
    if not sessions_dir.exists():
        print(f"  No sessions directory found")
        return {}
    
    # Get all session files sorted by name (which includes timestamp)
    session_files = sorted(sessions_dir.glob("*.jsonl"))
    print(f"  Found {len(session_files)} session files")
    
    if not session_files:
        return {}
    
    calculator = BehavioralCategoryCalculator()
    merged_profile = {}
    
    for session_file in session_files:
        counters = load_session_counters(session_file)
        
        if not counters:
            continue
        
        # Compute profile from this session's counters
        profile = calculator.compute_profile(counters)
        profile['last_updated'] = datetime.now().isoformat()
        profile['session_id'] = session_file.stem
        profile['user_id'] = user_id
        
        # Merge with accumulated profile
        if merged_profile:
            merged_profile = calculator.merge_profiles(merged_profile, profile)
        else:
            merged_profile = profile
    
    return merged_profile


def main():
    """Recompute profiles for all users."""
    print("=" * 60)
    print("Recomputing Behavioral Profiles from Session Logs")
    print("=" * 60)
    
    # Find all user directories
    user_dirs = [d for d in BASE_PATH.iterdir() if d.is_dir() and (d / "sessions").exists()]
    
    print(f"\nFound {len(user_dirs)} users with session data")
    
    for user_dir in sorted(user_dirs):
        user_id = user_dir.name
        print(f"\n📊 Processing {user_id}...")
        
        profile = recompute_profile_for_user(user_id)
        
        if not profile:
            print(f"  ⚠️ No valid profile computed")
            continue
        
        # Print summary
        categories = profile.get("categories", {})
        print(f"  Categories:")
        print(f"    AI Dependency: {categories.get('ai_dependency_level', 'unknown')}")
        print(f"    Speed Profile: {categories.get('speed_profile', 'unknown')}")
        print(f"    Experience:    {categories.get('experience_level', 'unknown')}")
        print(f"    Modality:      {categories.get('modality_expertise', 'unknown')}")
        print(f"    VLM Usage:     {categories.get('vlm_usage_pattern', 'unknown')}")
        
        # Primary tool
        primary_tool = profile.get("metrics", {}).get("primary_tool", {})
        print(f"    Primary Tool:  {primary_tool.get('primary_tool', 'unknown')}")
        
        # Save the profile
        profile_path = user_dir / "profile.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved profile to {profile_path.name}")
    
    print("\n" + "=" * 60)
    print("✅ Profile recomputation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
