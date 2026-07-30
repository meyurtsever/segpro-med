"""Shared label definitions used by the Label Manager and image annotators."""

from copy import deepcopy


DEFAULT_LABELS = (
    {
        "id": 1,
        "name": "Normal Tissue",
        "color": "#00ff00",
        "active": True,
        "predefined": True,
    },
    {
        "id": 2,
        "name": "Tumor",
        "color": "#ff0000",
        "active": True,
        "predefined": True,
    },
    {
        "id": 3,
        "name": "Organ",
        "color": "#0000ff",
        "active": True,
        "predefined": True,
    },
    {
        "id": 4,
        "name": "Lesion",
        "color": "#ffff00",
        "active": True,
        "predefined": True,
    },
    {
        "id": 5,
        "name": "ROI",
        "color": "#ff00ff",
        "active": True,
        "predefined": True,
    },
    {
        "id": 6,
        "name": "Other",
        "color": "#00ffff",
        "active": True,
        "predefined": True,
    },
)


def get_default_labels():
    """Return a mutable copy of the predefined image annotator labels."""
    return deepcopy(list(DEFAULT_LABELS))
