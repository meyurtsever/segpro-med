"""
User Preferences Manager for SegMed-Pro
Handles user-specific settings like modal display preferences.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class UserPreferencesManager:
    """Manages user preferences including modal display settings"""
    
    def __init__(self, preferences_file_path="db/user_preferences.json"):
        self.preferences_file_path = preferences_file_path
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Load user preferences from JSON file"""
        if not os.path.exists(self.preferences_file_path):
            logger.info(f"Preferences file not found, creating new: {self.preferences_file_path}")
            return {}
        
        try:
            with open(self.preferences_file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
            logger.info(f"Loaded preferences for {len(preferences)} users")
            return preferences
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            return {}
    
    def _save_preferences(self) -> bool:
        """Save preferences to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.preferences_file_path), exist_ok=True)
            
            with open(self.preferences_file_path, 'w', encoding='utf-8') as file:
                json.dump(self.preferences, file, indent=2)
            logger.info(f"Preferences saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get preferences for a specific user"""
        if user_id not in self.preferences:
            # Initialize default preferences for new user
            self.preferences[user_id] = {
                'modals': {
                    'show_editor_welcome': True,
                    'show_segmentation_complete': True,
                    'show_point_prompt_tip': True,
                    'show_box_prompt_tip': True
                },
                'created_at': self._get_timestamp()
            }
            self._save_preferences()
        
        return self.preferences[user_id]
    
    def update_modal_preference(self, user_id: str, modal_name: str, show: bool) -> bool:
        """Update modal display preference for a user"""
        try:
            user_prefs = self.get_user_preferences(user_id)
            
            if 'modals' not in user_prefs:
                user_prefs['modals'] = {}
            
            user_prefs['modals'][modal_name] = show
            user_prefs['last_updated'] = self._get_timestamp()
            
            self.preferences[user_id] = user_prefs
            return self._save_preferences()
        except Exception as e:
            logger.error(f"Error updating modal preference: {e}")
            return False
    
    def should_show_modal(self, user_id: str, modal_name: str) -> bool:
        """Check if a modal should be shown to a user"""
        user_prefs = self.get_user_preferences(user_id)
        return user_prefs.get('modals', {}).get(modal_name, True)
    
    def reset_user_preferences(self, user_id: str) -> bool:
        """Reset all preferences for a user to defaults"""
        try:
            if user_id in self.preferences:
                del self.preferences[user_id]
                return self._save_preferences()
            return True
        except Exception as e:
            logger.error(f"Error resetting preferences: {e}")
            return False
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Convenience methods for specific modals
    def should_show_editor_welcome(self, user_id: str) -> bool:
        """Check if editor welcome modal should be shown"""
        return self.should_show_modal(user_id, 'show_editor_welcome')
    
    def should_show_segmentation_complete(self, user_id: str) -> bool:
        """Check if segmentation complete modal should be shown"""
        return self.should_show_modal(user_id, 'show_segmentation_complete')
    
    def hide_editor_welcome(self, user_id: str) -> bool:
        """Set preference to hide editor welcome modal"""
        return self.update_modal_preference(user_id, 'show_editor_welcome', False)
    
    def hide_segmentation_complete(self, user_id: str) -> bool:
        """Set preference to hide segmentation complete modal"""
        return self.update_modal_preference(user_id, 'show_segmentation_complete', False)
    
    def should_show_point_prompt_tip(self, user_id: str) -> bool:
        """Check if point prompt tip modal should be shown"""
        return self.should_show_modal(user_id, 'show_point_prompt_tip')
    
    def should_show_box_prompt_tip(self, user_id: str) -> bool:
        """Check if box prompt tip modal should be shown"""
        return self.should_show_modal(user_id, 'show_box_prompt_tip')
    
    def hide_point_prompt_tip(self, user_id: str) -> bool:
        """Set preference to hide point prompt tip modal"""
        return self.update_modal_preference(user_id, 'show_point_prompt_tip', False)
    
    def hide_box_prompt_tip(self, user_id: str) -> bool:
        """Set preference to hide box prompt tip modal"""
        return self.update_modal_preference(user_id, 'show_box_prompt_tip', False)

# Global instance for easy import
_preferences_manager = None

def get_preferences_manager(preferences_file_path="db/user_preferences.json"):
    """Get or create global preferences manager instance"""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = UserPreferencesManager(preferences_file_path)
    return _preferences_manager
