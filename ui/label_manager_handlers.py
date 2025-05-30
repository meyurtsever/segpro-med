"""
SegMed-Pro Label Manager Handlers

This module contains event handlers for the label manager tab.
"""

import os
import re
from typing import List, Tuple, Optional

class LabelManagerHandlers:
    def __init__(self, state=None):
        self.state = state
        self.current_labels = []  # Internal storage with full RGB data [ID, Name, Color Preview, R, G, B]

    def load_label_set(self, file_obj):
        """Load ITK-SNAP label file and return table data with color preview"""
        if file_obj is None:
            return [], "No file selected."
        
        try:
            # Gradio File returns a dict with 'name' key
            file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
            
            labels = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse ITK-SNAP format: IDX R G B A VIS MSH "LABEL"
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Use regex to parse the line properly
                    match = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+\d+\s+\d+\s+"([^"]*)"', line)
                    if match:
                        idx, r, g, b, name = match.groups()
                        idx, r, g, b = int(idx), int(r), int(g), int(b)
                        
                        # Create accurate color preview with HTML
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        color_preview = f"<div style='width: 20px; height: 20px; background-color: {hex_color}; border: 1px solid #000;'></div>"
                        # Store both preview and RGB values for saving
                        labels.append([idx, name, color_preview, r, g, b])
            
            # Sort by index
            labels.sort(key=lambda x: x[0])
            self.current_labels = labels
            
            # Return only display columns for the table UI
            display_data = self.get_table_display_data(labels)
            return display_data, f"[OK] Loaded {len(labels)} labels successfully."
        except Exception as e:
            return [], f"[ERROR] Error loading file: {str(e)}"

    def sync_table_to_internal(self, table_data):
        """Sync table display data back to internal storage when user edits the table"""
        # Check if table_data is empty or None safely to avoid DataFrame ambiguity
        if table_data is None:
            return
        if hasattr(table_data, 'empty') and table_data.empty:
            return
        if isinstance(table_data, list) and len(table_data) == 0:
            return
        if not self.current_labels:
            return
        
        try:
            # Handle DataFrame or list for table_data
            if hasattr(table_data, 'values'):
                display_data = table_data.values.tolist()
            else:
                display_data = table_data if isinstance(table_data, list) else []
            
            # Update names in internal storage based on display data changes
            for i, display_row in enumerate(display_data):
                if i < len(self.current_labels) and len(display_row) >= 2:
                    # Update the label name (column 1) if it was edited
                    self.current_labels[i][1] = str(display_row[1])
                    
        except Exception as e:
            print(f"[DEBUG] Error syncing table data: {str(e)}")

    def save_label_set(self, table_data, file_name="label_set_edited.label"):
        """Save label data to ITK-SNAP format (Quick Save - overwrites default file)"""
        # Sync table edits back to internal storage first
        self.sync_table_to_internal(table_data)
        
        # Use internal full data instead of display table data
        if not self.current_labels:
            return None, "[ERROR] No label data to save."
        
        try:
            save_path = os.path.join(os.getcwd(), file_name)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                # Write ITK-SNAP header
                f.write("################################################\n")
                f.write("# ITK-SnAP Label Description File\n")
                f.write("# File format: \n")
                f.write("# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n")
                f.write("# Fields: \n")
                f.write("#    IDX:   Zero-based index \n")
                f.write("#    -R-:   Red color component (0..255)\n")
                f.write("#    -G-:   Green color component (0..255)\n")
                f.write("#    -B-:   Blue color component (0..255)\n")
                f.write("#    -A-:   Label transparency (0.00 .. 1.00)\n")
                f.write("#    VIS:   Label visibility (0 or 1)\n")
                f.write("#    IDX:   Label mesh visibility (0 or 1)\n")
                f.write("#  LABEL:   Label description \n")
                f.write("################################################\n")
                
                # Write label data from internal storage
                for row in self.current_labels:
                    idx = int(row[0])
                    name = str(row[1])
                    # Get RGB values from stored columns or extract from color preview
                    if len(row) >= 6:  # RGB values are stored in columns 3,4,5
                        r = int(row[3])
                        g = int(row[4]) 
                        b = int(row[5])
                    else:
                        # Extract RGB from color preview HTML (fallback)
                        color_preview = str(row[2])
                        color_match = re.search(r'background-color:\s*#([0-9a-fA-F]{6})', color_preview)
                        if color_match:
                            hex_color = color_match.group(1)
                            r = int(hex_color[0:2], 16)
                            g = int(hex_color[2:4], 16)
                            b = int(hex_color[4:6], 16)
                        else:
                            r = g = b = 255  # Default to white if color extraction fails
                    
                    # Format: IDX R G B A VIS MSH "LABEL"
                    f.write(f"    {idx:2d}   {r:3d}  {g:3d}  {b:3d}        1  1  1    \"{name}\"\n")
            
            return save_path, f"[OK] Saved to {save_path}"
            
        except Exception as e:
            return None, f"[ERROR] Error saving: {str(e)}"

    def save_label_set_with_filename(self, table_data, custom_filename):
        """Save label data to ITK-SNAP format with custom filename (Save Labels - new file)"""
        # Sync table edits back to internal storage first
        self.sync_table_to_internal(table_data)
        
        if not custom_filename or not custom_filename.strip():
            custom_filename = "label_set_custom.label"
        
        # Ensure .label extension
        if not custom_filename.endswith('.label'):
            custom_filename += '.label'
            
        return self.save_label_set(table_data, custom_filename)

    def add_new_label(self, table_data, new_name, color_hex):
        """Add a new label with specified name and color"""
        if not new_name or not new_name.strip():
            return table_data, "[ERROR] Please enter a label name."

        try:
            print(f"[DEBUG] color_hex received: {repr(color_hex)}")  # Debug print
            if not color_hex or not isinstance(color_hex, str):
                return table_data, "[ERROR] Invalid color format."
            color_hex = color_hex.strip()
            r = g = b = None
            # Handle #RRGGBB
            if len(color_hex) == 7 and color_hex.startswith('#') and all(c in '0123456789abcdefABCDEF' for c in color_hex[1:]):
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)
            # Handle rgba(r, g, b, a)
            elif color_hex.startswith('rgba'):
                match = re.match(r'rgba\(([^,]+),([^,]+),([^,]+),', color_hex)
                if match:
                    r = int(float(match.group(1)))
                    g = int(float(match.group(2)))
                    b = int(float(match.group(3)))
                else:
                    return table_data, "[ERROR] Invalid RGBA color format."
            else:
                return table_data, "[ERROR] Invalid color format. Please select a color using the color picker."

            # Handle DataFrame or list for table_data
            if hasattr(table_data, 'values'):
                data_list = table_data.values.tolist()
            else:
                data_list = table_data if table_data is not None else []

            # Find next available index from internal data
            existing_indices = [int(row[0]) for row in self.current_labels] if self.current_labels else []
            next_idx = max(existing_indices) + 1 if existing_indices else 1

            # Create accurate color preview with HTML
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            color_preview = f"<div style='width: 20px; height: 20px; background-color: {hex_color}; border: 1px solid #000;'></div>"
            
            # Add new label with RGB values stored for saving
            new_label = [next_idx, new_name.strip(), color_preview, r, g, b]
            self.current_labels.append(new_label)

            # Return only display columns for the table UI
            display_data = self.get_table_display_data(self.current_labels)
            return display_data, f"[OK] Added label '{new_name}' with ID {next_idx}."

        except Exception as e:
            return table_data, f"[ERROR] Error adding label: {str(e)}"

    def delete_label(self, table_data, label_idx):
        """Delete label by index"""
        # Handle empty table data
        if not self.current_labels:
            return table_data, "[ERROR] No labels to delete."
        
        try:
            label_idx = int(label_idx)
            
            original_count = len(self.current_labels)
            self.current_labels = [row for row in self.current_labels if int(row[0]) != label_idx]
            
            if len(self.current_labels) == original_count:
                return table_data, f"[ERROR] Label with ID {label_idx} not found."
            
            # Return only display columns for the table UI
            display_data = self.get_table_display_data(self.current_labels)
            return display_data, f"[OK] Deleted label with ID {label_idx}."
            
        except Exception as e:
            return table_data, f"[ERROR] Error deleting label: {str(e)}"

    def delete_label_by_name(self, table_data, label_name):
        """Delete label by name (for use with dropdown)."""
        if not label_name:
            return table_data, "[ERROR] Please select a label name to delete."
        # Handle empty table data
        if not self.current_labels:
            return table_data, "[ERROR] No labels to delete."
        try:
            # Work with the full internal data
            original_count = len(self.current_labels)
            self.current_labels = [row for row in self.current_labels if row[1] != label_name]
            if len(self.current_labels) == original_count:
                return table_data, f"[ERROR] Label with name '{label_name}' not found."
            
            # Return only display columns for the table UI
            display_data = self.get_table_display_data(self.current_labels)
            return display_data, f"[OK] Deleted label with name '{label_name}'."
        except Exception as e:
            return table_data, f"[ERROR] Error deleting label: {str(e)}"

    def get_label_names(self):
        """Return a list of label names for populating the delete dropdown."""
        return [row[1] for row in self.current_labels] if self.current_labels else []

    def get_table_display_data(self, full_data):
        """Return only the display columns (first 3) for the table UI"""
        if not full_data:
            return []
        return [[row[0], row[1], row[2]] for row in full_data]
