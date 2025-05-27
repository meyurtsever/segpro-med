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
        self.current_labels = []

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
                        
                        # labels.append([idx, name, color_preview, r, g, b])
                        # let's ignore r, g, b for now
                        labels.append([idx, name, color_preview])
            
            # Sort by index
            labels.sort(key=lambda x: x[0])
            self.current_labels = labels
            
            return labels, f"[OK] Loaded {len(labels)} labels successfully."
            
        except Exception as e:
            return [], f"[ERROR] Error loading file: {str(e)}"

    def save_label_set(self, table_data, file_name="label_set_edited.label"):
        """Save label data to ITK-SNAP format"""
        if not table_data or not isinstance(table_data, list):
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
                
                # Write label data
                for row in table_data:
                    idx = int(row[0])
                    name = str(row[1])
                    r = int(row[3])  # Skip color preview column
                    g = int(row[4])
                    b = int(row[5])
                    
                    # Format: IDX R G B A VIS MSH "LABEL"
                    f.write(f"    {idx:2d}   {r:3d}  {g:3d}  {b:3d}        1  1  1    \"{name}\"\n")
            
            self.current_labels = table_data
            return save_path, f"[OK] Saved to {save_path}"
            
        except Exception as e:
            return None, f"[ERROR] Error saving: {str(e)}"

    def add_new_label(self, table_data, new_name, color_hex):
        """Add a new label with specified name and color"""
        if not new_name or not new_name.strip():
            return table_data, "[ERROR] Please enter a label name."
        
        try:
            # Parse hex color
            if not color_hex.startswith('#') or len(color_hex) != 7:
                return table_data, "[ERROR] Invalid color format."
            
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            
            # Find next available index
            existing_indices = [int(row[0]) for row in table_data] if table_data else []
            next_idx = max(existing_indices) + 1 if existing_indices else 1
            
            # Create accurate color preview with hex value
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            color_preview = f"[*] {hex_color}"
            
            # Add new label
            new_label = [next_idx, new_name.strip(), color_preview, r, g, b]
            updated_table = table_data + [new_label] if table_data else [new_label]
            
            self.current_labels = updated_table
            return updated_table, f"[OK] Added label '{new_name}' with ID {next_idx}."
            
        except Exception as e:
            return table_data, f"[ERROR] Error adding label: {str(e)}"

    def delete_label(self, table_data, label_idx):
        """Delete label by index"""
        # Handle empty table data
        if table_data is None or (hasattr(table_data, 'empty') and table_data.empty) or (isinstance(table_data, list) and len(table_data) == 0):
            return table_data, "[ERROR] No labels to delete."
        
        try:
            label_idx = int(label_idx)
            
            # Convert DataFrame to list if needed
            if hasattr(table_data, 'values'):
                # It's a DataFrame
                data_list = table_data.values.tolist()
            else:
                # It's already a list
                data_list = table_data
            
            original_count = len(data_list)
            updated_table = [row for row in data_list if int(row[0]) != label_idx]
            
            if len(updated_table) == original_count:
                return table_data, f"[ERROR] Label with ID {label_idx} not found."
            
            self.current_labels = updated_table
            return updated_table, f"[OK] Deleted label with ID {label_idx}."
            
        except Exception as e:
            return table_data, f"[ERROR] Error deleting label: {str(e)}"

# Ensure Gradio table is configured to render HTML for the color_preview column
# Example Gradio table setup:
# import gradio as gr
# table = gr.DataFrame(headers=["Index", "Name", "Color Preview", "R", "G", "B"],
#                      datatype=["number", "str", "html", "number", "number", "number"],
#                      interactive=True)
