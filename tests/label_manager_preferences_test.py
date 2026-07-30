"""Regression tests for per-user Label Manager preferences."""

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auth.user_preferences import UserPreferencesManager
from ui.label_manager_handlers import LabelManagerHandlers


def _row_by_name(rows, name):
    return next(row for row in rows if row[1] == name)


class LabelManagerPreferencesTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        preferences_path = (
            Path(self.temp_directory.name) / "user_preferences.json"
        )
        self.preferences = UserPreferencesManager(str(preferences_path))
        self.handler = LabelManagerHandlers(
            preferences_manager=self.preferences
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_predefined_and_custom_labels_are_private_per_user(self):
        alice_rows, _, _ = self.handler.load_user_labels("alice")
        self.assertTrue(_row_by_name(alice_rows, "Tumor"))
        self.assertTrue(_row_by_name(alice_rows, "Organ"))

        alice_rows, _, _ = self.handler.add_new_label(
            alice_rows, "Kidney", "#123456", "alice"
        )
        _row_by_name(alice_rows, "Tumor")[3] = False
        alice_rows, _, alice_config = self.handler.sync_table_to_user(
            alice_rows, "alice"
        )

        active_names = alice_config["label_list"]
        self.assertIn("Kidney", active_names)
        self.assertNotIn("Tumor", active_names)
        self.assertFalse(alice_config["use_default_label"])

        bob_rows, _, bob_config = self.handler.load_user_labels("bob")
        self.assertFalse(any(row[1] == "Kidney" for row in bob_rows))
        self.assertTrue(_row_by_name(bob_rows, "Tumor")[3])
        self.assertIn("Tumor", bob_config["label_list"])
        self.assertNotEqual(
            self.preferences.get_user_labels("alice"),
            self.preferences.get_user_labels("bob"),
        )

    def test_predefined_labels_are_made_passive_instead_of_deleted(self):
        rows, _, _ = self.handler.load_user_labels("alice")
        rows, status, _ = self.handler.delete_label_by_name(
            rows, "Tumor", "alice"
        )

        self.assertIn("cannot be deleted", status)
        self.assertTrue(_row_by_name(rows, "Tumor"))

    def test_loaded_set_keeps_predefined_and_persists_custom_labels(self):
        label_file = Path(self.temp_directory.name) / "custom.label"
        label_file.write_text(
            '1 255 0 0 1 1 1 "Tumor"\n'
            '9 18 52 86 1 1 1 "Custom Region"\n',
            encoding="utf-8",
        )

        rows, status, config = self.handler.load_label_set(
            str(label_file), "alice"
        )

        self.assertTrue(status.startswith("[OK]"))
        self.assertTrue(_row_by_name(rows, "Tumor"))
        self.assertTrue(_row_by_name(rows, "Organ"))
        self.assertEqual(_row_by_name(rows, "Custom Region")[0], 9)
        self.assertIn("Custom Region", config["label_list"])

        reloaded_rows, _, _ = self.handler.load_user_labels("alice")
        self.assertTrue(_row_by_name(reloaded_rows, "Custom Region"))

    def test_name_and_color_edits_persist_per_user(self):
        rows, _, _ = self.handler.load_user_labels("alice")
        tumor = _row_by_name(rows, "Tumor")
        tumor[1] = "Mass"
        tumor[2] = "#abcdef"

        rows, status, config = self.handler.sync_table_to_user(
            rows, "alice"
        )

        self.assertEqual(status, "[OK] Label preferences saved.")
        renamed = _row_by_name(rows, "Mass")
        self.assertEqual(renamed[0], 2)
        self.assertEqual(renamed[2], "#abcdef")
        self.assertIn("Mass", config["label_list"])
        self.assertNotIn("Tumor", config["label_list"])
        mass_index = config["label_list"].index("Mass")
        self.assertEqual(config["label_colors"][mass_index], "#abcdef")

        reloaded_rows, _, _ = self.handler.load_user_labels("alice")
        self.assertEqual(_row_by_name(reloaded_rows, "Mass")[2], "#abcdef")

        bob_rows, _, _ = self.handler.load_user_labels("bob")
        self.assertTrue(_row_by_name(bob_rows, "Tumor"))
        self.assertFalse(any(row[1] == "Mass" for row in bob_rows))

    def test_quick_add_row_becomes_a_persisted_custom_label(self):
        rows, _, _ = self.handler.load_user_labels("alice")
        quick_add = _row_by_name(rows, "+")
        quick_add[1] = "Quick Label"
        quick_add[2] = "#445566"

        saved_rows, status, config = self.handler.sync_table_to_user(
            rows, "alice"
        )
        self.assertEqual(status, "[OK] Label preferences saved.")
        quick_label = _row_by_name(saved_rows, "Quick Label")
        self.assertGreater(quick_label[0], 0)
        self.assertEqual(quick_label[2], "#445566")
        self.assertIn("Quick Label", config["label_list"])
        self.assertTrue(_row_by_name(saved_rows, "+"))

        reloaded_rows, _, _ = self.handler.load_user_labels("alice")
        self.assertEqual(
            _row_by_name(reloaded_rows, "Quick Label")[2], "#445566"
        )
        self.assertTrue(_row_by_name(reloaded_rows, "+"))


if __name__ == "__main__":
    unittest.main()
