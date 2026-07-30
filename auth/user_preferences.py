"""
User Preferences Manager for SegMed-Pro
Handles user-specific settings like modal display preferences.
"""

import json
import logging
import os
import tempfile
import threading
from copy import deepcopy
from typing import Any, Dict

from utils.label_config import get_default_labels


logger = logging.getLogger(__name__)


class UserPreferencesManager:
    """Manages user preferences including modal and label settings."""

    def __init__(self, preferences_file_path="db/user_preferences.json"):
        self.preferences_file_path = preferences_file_path
        self._lock = threading.RLock()
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Load user preferences from JSON file."""
        if not os.path.exists(self.preferences_file_path):
            logger.info(
                "Preferences file not found, creating new: %s",
                self.preferences_file_path,
            )
            return {}

        try:
            with open(self.preferences_file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
            logger.info("Loaded preferences for %s users", len(preferences))
            return preferences
        except Exception as e:
            logger.error("Error loading preferences: %s", e)
            return {}

    def _save_preferences(self) -> bool:
        """Atomically save preferences to the JSON file."""
        temp_path = None
        try:
            with self._lock:
                directory = os.path.dirname(os.path.abspath(self.preferences_file_path))
                os.makedirs(directory, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    dir=directory,
                    delete=False,
                    suffix='.tmp',
                ) as temp_file:
                    temp_path = temp_file.name
                    json.dump(self.preferences, temp_file, indent=2)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, self.preferences_file_path)
            logger.info("Preferences saved successfully")
            return True
        except Exception as e:
            logger.error("Error saving preferences: %s", e)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return False

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get preferences for a specific user."""
        with self._lock:
            if user_id not in self.preferences:
                self.preferences[user_id] = {
                    'modals': {
                        'show_editor_welcome': True,
                        'show_segmentation_complete': True,
                        'show_point_prompt_tip': True,
                        'show_box_prompt_tip': True,
                    },
                    'created_at': self._get_timestamp(),
                }
                self._save_preferences()

            return self.preferences[user_id]

    def get_user_labels(self, user_id: str):
        """Return one user's labels, initializing the predefined set."""
        with self._lock:
            user_prefs = self.get_user_preferences(user_id)
            labels = user_prefs.get('label_manager', {}).get('labels')
            if not isinstance(labels, list):
                labels = get_default_labels()
                user_prefs['label_manager'] = {'labels': labels}
                user_prefs['last_updated'] = self._get_timestamp()
                self.preferences[user_id] = user_prefs
                self._save_preferences()
            return deepcopy(labels)

    def update_user_labels(self, user_id: str, labels) -> bool:
        """Persist Label Manager rows for one user only."""
        try:
            with self._lock:
                user_prefs = self.get_user_preferences(user_id)
                user_prefs['label_manager'] = {'labels': deepcopy(labels)}
                user_prefs['last_updated'] = self._get_timestamp()
                self.preferences[user_id] = user_prefs
                return self._save_preferences()
        except Exception as e:
            logger.error("Error updating label preferences: %s", e)
            return False

    def update_modal_preference(
        self, user_id: str, modal_name: str, show: bool
    ) -> bool:
        """Update modal display preference for a user."""
        try:
            with self._lock:
                user_prefs = self.get_user_preferences(user_id)
                if 'modals' not in user_prefs:
                    user_prefs['modals'] = {}

                user_prefs['modals'][modal_name] = show
                user_prefs['last_updated'] = self._get_timestamp()
                self.preferences[user_id] = user_prefs
                return self._save_preferences()
        except Exception as e:
            logger.error("Error updating modal preference: %s", e)
            return False

    def should_show_modal(self, user_id: str, modal_name: str) -> bool:
        """Check if a modal should be shown to a user."""
        user_prefs = self.get_user_preferences(user_id)
        return user_prefs.get('modals', {}).get(modal_name, True)

    def reset_user_preferences(self, user_id: str) -> bool:
        """Reset all preferences for a user to defaults."""
        try:
            with self._lock:
                if user_id in self.preferences:
                    del self.preferences[user_id]
                    return self._save_preferences()
                return True
        except Exception as e:
            logger.error("Error resetting preferences: %s", e)
            return False

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp as string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def should_show_editor_welcome(self, user_id: str) -> bool:
        """Check if editor welcome modal should be shown."""
        return self.should_show_modal(user_id, 'show_editor_welcome')

    def should_show_segmentation_complete(self, user_id: str) -> bool:
        """Check if segmentation complete modal should be shown."""
        return self.should_show_modal(user_id, 'show_segmentation_complete')

    def hide_editor_welcome(self, user_id: str) -> bool:
        """Set preference to hide editor welcome modal."""
        return self.update_modal_preference(
            user_id, 'show_editor_welcome', False
        )

    def hide_segmentation_complete(self, user_id: str) -> bool:
        """Set preference to hide segmentation complete modal."""
        return self.update_modal_preference(
            user_id, 'show_segmentation_complete', False
        )

    def should_show_point_prompt_tip(self, user_id: str) -> bool:
        """Check if point prompt tip modal should be shown."""
        return self.should_show_modal(user_id, 'show_point_prompt_tip')

    def should_show_box_prompt_tip(self, user_id: str) -> bool:
        """Check if box prompt tip modal should be shown."""
        return self.should_show_modal(user_id, 'show_box_prompt_tip')

    def hide_point_prompt_tip(self, user_id: str) -> bool:
        """Set preference to hide point prompt tip modal."""
        return self.update_modal_preference(
            user_id, 'show_point_prompt_tip', False
        )

    def hide_box_prompt_tip(self, user_id: str) -> bool:
        """Set preference to hide box prompt tip modal."""
        return self.update_modal_preference(
            user_id, 'show_box_prompt_tip', False
        )


_preferences_manager = None


def get_preferences_manager(preferences_file_path="db/user_preferences.json"):
    """Get or create global preferences manager instance."""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = UserPreferencesManager(preferences_file_path)
    return _preferences_manager
