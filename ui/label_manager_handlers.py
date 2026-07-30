"""Event handlers and persistence helpers for the Label Manager tab."""

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from auth.user_preferences import get_preferences_manager
from utils.label_config import get_default_labels


class LabelManagerHandlers:
    """Manage predefined and custom labels without sharing state across users."""

    QUICK_ADD_NAME = '+'

    def __init__(self, state=None, preferences_manager=None):
        self.state = state
        self.preferences_manager = (
            preferences_manager or get_preferences_manager()
        )

    @staticmethod
    def resolve_user_id(user_state) -> str:
        """Resolve a stable user identifier from a Gradio state value."""
        if isinstance(user_state, dict):
            user_id = user_state.get('user_id')
            if user_id:
                return str(user_id)
        if isinstance(user_state, str) and user_state.strip():
            return user_state.strip()
        return 'guest'

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {
                '1', 'true', 'yes', 'on', 'active'
            }
        return bool(value)

    @staticmethod
    def _normalize_color(value: Any, fallback: str = '#808080') -> str:
        text = str(value or '').strip()
        direct_match = re.fullmatch(r'#([0-9a-fA-F]{6})', text)
        if direct_match:
            return f"#{direct_match.group(1).lower()}"

        preview_match = re.search(
            r'background-color:\s*#([0-9a-fA-F]{6})', text
        )
        if preview_match:
            return f"#{preview_match.group(1).lower()}"
        return fallback

    @staticmethod
    def _color_preview(color: str) -> str:
        return (
            "<div style='width: 20px; height: 20px; "
            f"background-color: {color}; border: 1px solid #000;'></div>"
        )

    @staticmethod
    def _table_rows(table_data) -> List[List[Any]]:
        if table_data is None:
            return []
        if hasattr(table_data, 'values'):
            return table_data.values.tolist()
        if isinstance(table_data, list):
            return table_data
        return []

    def _normalize_labels(
        self, labels: Optional[Iterable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        defaults = get_default_labels()
        defaults_by_id = {label['id']: label for label in defaults}
        defaults_by_name = {
            label['name'].casefold(): label for label in defaults
        }
        overrides_by_default_id = {}
        custom_candidates = []

        for raw_label in labels or []:
            if not isinstance(raw_label, dict):
                continue
            try:
                label_id = int(raw_label.get('id', 0))
            except (TypeError, ValueError):
                label_id = 0
            name = str(raw_label.get('name', '')).strip()
            default = defaults_by_id.get(label_id)
            if default is None and name:
                default = defaults_by_name.get(name.casefold())

            if default is not None:
                overrides_by_default_id[default['id']] = {
                    'id': default['id'],
                    'name': name or default['name'],
                    'color': self._normalize_color(
                        raw_label.get('color'), default['color']
                    ),
                    'active': self._to_bool(
                        raw_label.get('active', True)
                    ),
                    'predefined': True,
                }
                continue
            if name:
                custom_candidates.append(raw_label)

        normalized = []
        used_ids = set()
        used_names = set()
        for default in defaults:
            label = dict(
                overrides_by_default_id.get(default['id'], default)
            )
            label['name'] = (
                str(label.get('name', '')).strip() or default['name']
            )
            label['color'] = self._normalize_color(
                label.get('color'), default['color']
            )
            label['active'] = self._to_bool(
                label.get('active', True)
            )
            label['predefined'] = True
            normalized.append(label)
            used_ids.add(label['id'])
            used_names.add(label['name'].casefold())

        next_id = max(used_ids, default=0) + 1
        for raw_label in custom_candidates:
            name = str(raw_label.get('name', '')).strip()
            name_key = name.casefold()
            if not name or name_key in used_names:
                continue
            try:
                requested_id = int(raw_label.get('id', 0))
            except (TypeError, ValueError):
                requested_id = 0
            if requested_id <= 0 or requested_id in used_ids:
                while next_id in used_ids:
                    next_id += 1
                requested_id = next_id
            used_ids.add(requested_id)
            used_names.add(name_key)
            next_id = max(next_id, requested_id + 1)
            normalized.append({
                'id': requested_id,
                'name': name,
                'color': self._normalize_color(raw_label.get('color')),
                'active': self._to_bool(raw_label.get('active', True)),
                'predefined': False,
            })

        return normalized

    def _get_user_labels(self, user_state) -> List[Dict[str, Any]]:
        user_id = self.resolve_user_id(user_state)
        stored = self.preferences_manager.get_user_labels(user_id)
        normalized = self._normalize_labels(stored)
        if normalized != stored:
            self.preferences_manager.update_user_labels(user_id, normalized)
        return normalized

    def _persist(
        self, user_state, labels: Iterable[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        user_id = self.resolve_user_id(user_state)
        normalized = self._normalize_labels(labels)
        saved = self.preferences_manager.update_user_labels(
            user_id, normalized
        )
        return normalized, saved

    def _labels_from_table(
        self, table_data, user_state
    ) -> List[Dict[str, Any]]:
        existing = self._get_user_labels(user_state)
        existing_by_id = {label['id']: label for label in existing}
        defaults_by_id = {
            label['id']: label for label in get_default_labels()
        }
        labels = []
        used_ids = set(existing_by_id)
        next_id = max(used_ids, default=0) + 1

        for row in self._table_rows(table_data):
            if len(row) < 2:
                continue
            try:
                label_id = int(row[0])
                if label_id <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                name = str(row[1] or '').strip()
                if not name or name == self.QUICK_ADD_NAME:
                    continue
                while next_id in used_ids:
                    next_id += 1
                label_id = next_id
                used_ids.add(label_id)
                next_id += 1

            default = defaults_by_id.get(label_id)
            previous = existing_by_id.get(label_id, {})
            fallback_name = previous.get(
                'name', default['name'] if default else ''
            )
            name = str(row[1] or '').strip() or fallback_name
            if not name:
                continue
            fallback_color = previous.get(
                'color', default['color'] if default else '#808080'
            )
            label = {
                'id': label_id,
                'name': name,
                'color': self._normalize_color(
                    row[2] if len(row) > 2 else None,
                    fallback_color,
                ),
                'active': self._to_bool(
                    row[3] if len(row) > 3 else previous.get(
                        'active', True
                    )
                ),
                'predefined': bool(
                    default is not None
                    or previous.get('predefined', False)
                ),
            }
            labels.append(label)

        present_ids = {label['id'] for label in labels}
        for default in get_default_labels():
            if default['id'] not in present_ids:
                restored = dict(
                    existing_by_id.get(default['id'], default)
                )
                restored['id'] = default['id']
                restored['predefined'] = True
                labels.append(restored)

        return self._normalize_labels(labels)

    def _pending_table_rows(self, table_data) -> List[List[Any]]:
        """Keep native add-row entries visible until a name is entered."""
        pending = []
        for row in self._table_rows(table_data):
            try:
                label_id = int(row[0])
                if label_id > 0:
                    continue
            except (TypeError, ValueError, IndexError):
                pass
            name = str(row[1] if len(row) > 1 else '').strip()
            if name and name != self.QUICK_ADD_NAME:
                continue
            color = self._normalize_color(
                row[2] if len(row) > 2 else None
            )
            active = self._to_bool(
                row[3] if len(row) > 3 else True
            )
            pending.append(['', self.QUICK_ADD_NAME, color, active])
        return pending

    def _quick_add_row(self) -> List[Any]:
        return ['', self.QUICK_ADD_NAME, '#808080', True]

    def get_table_display_data(
        self, full_data: Iterable[Dict[str, Any]]
    ) -> List[List[Any]]:
        """Return the four columns displayed by the Current Labels table."""
        return [
            [
                label['id'],
                label['name'],
                label['color'],
                label['active'],
            ]
            for label in full_data
        ]

    @staticmethod
    def get_annotator_config(
        labels: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build dynamic image_annotator choices from active labels only."""
        active_labels = [label for label in labels if label['active']]
        # Send raw names. The image_annotator Python component converts
        # them to ``(label, index)`` choices while serializing an update.
        # Pre-paired values are paired a second time and break the label modal.
        return {
            'label_list': [
                label['name'] for label in active_labels
            ],
            'label_colors': [
                label['color'] for label in active_labels
            ],
            'use_default_label': False,
        }

    def _ui_result(self, labels, status):
        return (
            self.get_table_display_data(labels) + [
                self._quick_add_row()
            ],
            status,
            self.get_annotator_config(labels),
        )

    def load_user_labels(self, user_state):
        """Load one user's private label table and active annotator choices."""
        labels = self._get_user_labels(user_state)
        user_id = self.resolve_user_id(user_state)
        return self._ui_result(
            labels, f"[OK] Loaded labels for {user_id}."
        )

    def get_default_ui_state(self):
        """Return a non-persistent default state for logout/reset rendering."""
        return self._ui_result(get_default_labels(), "")

    def sync_table_to_user(self, table_data, user_state):
        """Persist editable attributes and preserve the bottom quick-add row."""
        pending_rows = self._pending_table_rows(table_data)
        labels = self._labels_from_table(table_data, user_state)
        labels, saved = self._persist(user_state, labels)
        status = (
            "[OK] Label preferences saved."
            if saved
            else "[ERROR] Could not save label preferences."
        )
        return (
            self.get_table_display_data(labels) + (
                pending_rows or [self._quick_add_row()]
            ),
            status,
            self.get_annotator_config(labels),
        )

    def load_label_set(self, file_obj, user_state=None):
        """Load a label file as this user's custom set plus predefined labels."""
        if file_obj is None:
            labels = self._get_user_labels(user_state)
            return self._ui_result(labels, "No file selected.")

        try:
            file_path = (
                file_obj.name if hasattr(file_obj, 'name') else file_obj
            )
            imported = []
            with open(file_path, 'r', encoding='utf-8') as label_file:
                for line in label_file:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith('#')
                        or line.startswith('//')
                    ):
                        continue
                    match = re.match(
                        r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
                        r'\s+[\d.]+\s+\d+\s+\d+\s+"([^"]*)"',
                        line,
                    )
                    if match:
                        label_id, red, green, blue, name = match.groups()
                        imported.append({
                            'id': int(label_id),
                            'name': name.strip(),
                            'color': (
                                f"#{int(red):02x}{int(green):02x}"
                                f"{int(blue):02x}"
                            ),
                            'active': True,
                            'predefined': False,
                        })

            existing = self._get_user_labels(user_state)
            active_defaults = {
                label['id']: label['active']
                for label in existing
                if label['predefined']
            }
            labels = get_default_labels()
            for label in labels:
                label['active'] = active_defaults.get(label['id'], True)

            default_names = {
                label['name'].casefold() for label in labels
            }
            used_ids = {label['id'] for label in labels}
            next_id = max(used_ids) + 1
            for imported_label in imported:
                if imported_label['name'].casefold() in default_names:
                    continue
                requested_id = imported_label['id']
                if requested_id <= 0 or requested_id in used_ids:
                    while next_id in used_ids:
                        next_id += 1
                    requested_id = next_id
                imported_label['id'] = requested_id
                used_ids.add(requested_id)
                next_id = max(next_id, requested_id + 1)
                labels.append(imported_label)

            labels, saved = self._persist(user_state, labels)
            status = (
                f"[OK] Loaded {len(imported)} labels successfully."
                if saved
                else "[ERROR] Labels loaded but preferences could not be saved."
            )
            return self._ui_result(labels, status)
        except Exception as e:
            labels = self._get_user_labels(user_state)
            return self._ui_result(
                labels, f"[ERROR] Error loading file: {str(e)}"
            )

    def add_new_label(
        self, table_data, new_name, color_hex, user_state=None
    ):
        """Add and activate a custom label for the current user."""
        labels = self._labels_from_table(table_data, user_state)
        name = str(new_name or '').strip()
        if not name:
            return self._ui_result(
                labels, "[ERROR] Please enter a label name."
            )
        if any(label['name'].casefold() == name.casefold() for label in labels):
            return self._ui_result(
                labels, f"[ERROR] Label '{name}' already exists."
            )

        color = self._normalize_color(color_hex, '')
        if not color:
            return self._ui_result(
                labels,
                "[ERROR] Invalid color. Please use the color picker.",
            )

        next_id = max((label['id'] for label in labels), default=0) + 1
        labels.append({
            'id': next_id,
            'name': name,
            'color': color,
            'active': True,
            'predefined': False,
        })
        labels, saved = self._persist(user_state, labels)
        status = (
            f"[OK] Added label '{name}' with ID {next_id}."
            if saved
            else "[ERROR] Could not save the new label."
        )
        return self._ui_result(labels, status)

    def delete_label_by_name(
        self, table_data, label_name, user_state=None
    ):
        """Delete a custom label; predefined labels can only be passive."""
        labels = self._labels_from_table(table_data, user_state)
        if not label_name:
            return self._ui_result(
                labels, "[ERROR] Please select a label name to delete."
            )

        selected = next(
            (label for label in labels if label['name'] == label_name),
            None,
        )
        if selected is None:
            return self._ui_result(
                labels, f"[ERROR] Label '{label_name}' not found."
            )
        if selected['predefined']:
            return self._ui_result(
                labels,
                "[ERROR] Predefined labels cannot be deleted. "
                "Clear Active to make the label passive.",
            )

        labels = [
            label for label in labels if label['id'] != selected['id']
        ]
        labels, saved = self._persist(user_state, labels)
        status = (
            f"[OK] Deleted label '{label_name}'."
            if saved
            else "[ERROR] Could not save the deletion."
        )
        return self._ui_result(labels, status)

    def save_label_set(
        self,
        table_data,
        file_name="label_set_edited.label",
        user_state=None,
    ):
        """Save the current user's label table in ITK-SNAP format."""
        labels = self._labels_from_table(table_data, user_state)
        labels, saved = self._persist(user_state, labels)
        if not saved:
            return None, "[ERROR] Could not save label preferences."
        if not labels:
            return None, "[ERROR] No label data to save."

        try:
            safe_name = os.path.basename(file_name)
            save_path = os.path.join(os.getcwd(), safe_name)
            with open(save_path, 'w', encoding='utf-8') as label_file:
                label_file.write("################################################\n")
                label_file.write("# ITK-SnAP Label Description File\n")
                label_file.write("# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n")
                label_file.write("################################################\n")
                for label in labels:
                    color = label['color'].lstrip('#')
                    red = int(color[0:2], 16)
                    green = int(color[2:4], 16)
                    blue = int(color[4:6], 16)
                    visible = 1 if label['active'] else 0
                    name = label['name'].replace('"', "'")
                    label_file.write(
                        f"    {label['id']:2d}   {red:3d}  {green:3d}  "
                        f"{blue:3d}        1  {visible}  1    "
                        f"\"{name}\"\n"
                    )
            return save_path, f"[OK] Saved to {save_path}"
        except Exception as e:
            return None, f"[ERROR] Error saving: {str(e)}"

    def save_label_set_with_filename(
        self, table_data, custom_filename, user_state=None
    ):
        """Save labels with a user-provided .label filename."""
        filename = str(custom_filename or '').strip()
        if not filename:
            filename = "label_set_custom.label"
        if not filename.lower().endswith('.label'):
            filename += '.label'
        return self.save_label_set(table_data, filename, user_state)

    def get_label_names(self, user_state=None):
        """Return label names for the current user's delete dropdown."""
        return [
            label['name'] for label in self._get_user_labels(user_state)
        ]
