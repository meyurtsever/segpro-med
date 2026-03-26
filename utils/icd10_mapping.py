"""
ICD-10 Mapping for Brain MRI Datasets
======================================
Covers all diagnostic folder labels found in:
  • Gazi 2025  →  C:\\Gazi\\TR_TBP_Anonymised_enc\\Anonymised\\500 MR
  • Gazi 2020  →  local cvm_48_t1 / cvm_t1 sub-datasets

Each entry maps the uppercase folder name that lives on disk to:
  - icd10_code        : primary WHO ICD-10 code
  - icd10_description : human-readable description (with dataset label in parentheses)
  - category          : broad clinical grouping
  - dataset           : which dataset(s) carry this class
  - aliases           : list of lowercase search terms that should resolve to this folder
                        (includes the ICD code itself, so typing the code in the
                        search box triggers the correct folder filter)

Bidirectional search is supported:
  resolve_query_to_folders("cvm")    → ["CAVERNOMA"]
  resolve_query_to_folders("D18.02") → ["CAVERNOMA"]
  resolve_query_to_folders("glioblastoma") → ["HGG"]
  resolve_query_to_folders("D33.2")  → ["DERMOID_TUMOR", "EPIDERMOID_CYST"]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Primary ICD-10 Mapping Table
# key  = folder name as it appears on disk (UPPERCASE)
# ─────────────────────────────────────────────────────────────────────────────

ICD10_MAPPING: Dict[str, Dict] = {

    # ── Vascular Malformations ────────────────────────────────────────────────

    "AVM": {
        "icd10_code": "Q28.2",
        "icd10_description": "Arteriovenous malformation of cerebral vessels",
        "category": "Vascular Malformation",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "avm",
            "arteriovenous malformation",
            "cerebral avm",
            "brain avm",
            "intracranial avm",
            "q28.2",
            "q282",
        ],
    },

    "CAVERNOMA": {
        "icd10_code": "D18.02",
        "icd10_description": "Hemangioma of intracranial structures (Cavernous Malformation)",
        "category": "Vascular Malformation",
        "dataset": ["Gazi 2020", "Gazi 2025"],
        "aliases": [
            # folder aliases
            "cavernoma",
            "cvm",
            "cvm_48_t1",
            "cvm_t1",
            # clinical synonyms
            "cavernous malformation",
            "cavernous hemangioma",
            "cavernous angioma",
            "cerebral cavernoma",
            "cerebral cavernous malformation",
            "ccm",
            # ICD codes
            "d18.02",
            "d1802",
        ],
    },

    # ── Cystic Lesions ────────────────────────────────────────────────────────

    "CYST": {
        "icd10_code": "G93.0",
        "icd10_description": "Cerebral cysts",
        "category": "Cystic Lesion",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "cyst",
            "brain cyst",
            "arachnoid cyst",
            "cerebral cyst",
            "porencephalic cyst",
            "glioependymal cyst",
            "g93.0",
            "g930",
        ],
    },

    "EPIDERMOID_CYST": {
        "icd10_code": "D33.2",
        "icd10_description": "Benign neoplasm of brain, unspecified (Epidermoid Cyst)",
        "category": "Congenital Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "epidermoid",
            "epidermoid cyst",
            "intracranial epidermoid",
            "cholesteatoma",
            "pearly tumor",
            # ICD code shared with DERMOID_TUMOR; handled via reverse-lookup
        ],
    },

    # ── Congenital / Developmental Tumors ────────────────────────────────────

    "DERMOID_TUMOR": {
        "icd10_code": "D33.2",
        "icd10_description": "Benign neoplasm of brain, unspecified (Dermoid Tumor)",
        "category": "Congenital Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "dermoid",
            "dermoid tumor",
            "dermoid cyst",
            "intracranial dermoid",
            # ICD code (note: D33.2 also covers EPIDERMOID_CYST – both returned on code search)
            "d33.2",
        ],
    },

    # ── Germ Cell Tumors ──────────────────────────────────────────────────────

    "GERMINOMA": {
        "icd10_code": "C71.9",
        "icd10_description": "Malignant neoplasm of brain, unspecified (Germinoma)",
        "category": "Germ Cell Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "germinoma",
            "pineal germinoma",
            "intracranial germinoma",
            "suprasellar germinoma",
            "germ cell tumor",
            "gct",
            # Note: C71.9 is shared with HGG; both returned on code search
        ],
    },

    # ── Vascular Tumors ───────────────────────────────────────────────────────

    "HEMANGIOBLASTOMA": {
        "icd10_code": "D33.1",
        "icd10_description": "Benign neoplasm of brain, infratentorial (Hemangioblastoma)",
        "category": "Vascular Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "hemangioblastoma",
            "haemangioblastoma",
            "cerebellar hemangioblastoma",
            "von hippel-lindau",
            "vhl tumor",
            "d33.1",
            "d331",
        ],
    },

    # ── Cerebrovascular ───────────────────────────────────────────────────────

    "HEMORRHAGE": {
        "icd10_code": "I61.9",
        "icd10_description": "Nontraumatic intracerebral hemorrhage, unspecified",
        "category": "Cerebrovascular",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "hemorrhage",
            "haemorrhage",
            "bleed",
            "brain bleed",
            "intracerebral hemorrhage",
            "intracranial hemorrhage",
            "ich",
            "cerebral hemorrhage",
            "spontaneous hemorrhage",
            "i61.9",
            "i619",
        ],
    },

    "STROKE": {
        "icd10_code": "I63.9",
        "icd10_description": "Cerebral infarction, unspecified",
        "category": "Cerebrovascular",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "stroke",
            "infarction",
            "ischemic stroke",
            "ischaemic stroke",
            "cerebral infarction",
            "cerebral ischemia",
            "cva",
            "cerebrovascular accident",
            "i63.9",
            "i639",
        ],
    },

    # ── Gliomas ───────────────────────────────────────────────────────────────

    "HGG": {
        "icd10_code": "C71.9",
        "icd10_description": "Malignant neoplasm of brain, unspecified (High Grade Glioma)",
        "category": "Glioma",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "hgg",
            "high grade glioma",
            "glioblastoma",
            "gbm",
            "glioblastoma multiforme",
            "anaplastic astrocytoma",
            "anaplastic glioma",
            "grade 4",
            "grade iv",
            "grade 4 glioma",
            "grade iii glioma",
            "who grade 4",
            # ICD code (shared with GERMINOMA; both returned on code search)
            "c71.9",
            "c719",
        ],
    },

    "LGG": {
        "icd10_code": "D43.2",
        "icd10_description": "Neoplasm of uncertain behavior of brain, unspecified (Low Grade Glioma)",
        "category": "Glioma",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "lgg",
            "low grade glioma",
            "grade 2",
            "grade ii",
            "grade 2 glioma",
            "astrocytoma",
            "oligodendroglioma",
            "diffuse glioma",
            "idh mutant glioma",
            "d43.2",
            "d432",
        ],
    },

    # ── Lymphoma ──────────────────────────────────────────────────────────────

    "LYMPHOMA": {
        "icd10_code": "C85.90",
        "icd10_description": "Non-Hodgkin lymphoma, unspecified (Primary CNS Lymphoma)",
        "category": "Lymphoma",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "lymphoma",
            "pcnsl",
            "primary cns lymphoma",
            "cnsl",
            "brain lymphoma",
            "cns lymphoma",
            "non-hodgkin lymphoma",
            "nhl",
            "dlbcl brain",
            "c85.90",
            "c8590",
        ],
    },

    # ── Meningeal Tumors ──────────────────────────────────────────────────────

    "MENINGIOMA": {
        "icd10_code": "D32.0",
        "icd10_description": "Benign neoplasm of cerebral meninges",
        "category": "Meningeal Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "meningioma",
            "cerebral meningioma",
            "intracranial meningioma",
            "meningeal tumor",
            "who grade 1 meningioma",
            "d32.0",
            "d320",
        ],
    },

    # ── Secondary / Metastatic Neoplasms ──────────────────────────────────────

    "METASTASIS": {
        "icd10_code": "C79.31",
        "icd10_description": "Secondary malignant neoplasm of brain",
        "category": "Secondary Neoplasm",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "metastasis",
            "metastases",
            "brain met",
            "brain mets",
            "secondary brain tumor",
            "secondary malignancy",
            "brain metastasis",
            "brain secondaries",
            "c79.31",
            "c7931",
        ],
    },

    # ── Normal / Control ──────────────────────────────────────────────────────

    "NORMAL": {
        "icd10_code": "Z03.89",
        "icd10_description": "Encounter for observation for other suspected diseases and conditions ruled out",
        "category": "Normal / Control",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "normal",
            "healthy",
            "no pathology",
            "unremarkable",
            "control",
            "no finding",
            "z03.89",
            "z0389",
        ],
    },

    # ── Osseous Tumors ────────────────────────────────────────────────────────

    "OSTEOMA": {
        "icd10_code": "D16.4",
        "icd10_description": "Benign neoplasm of bones of skull and face",
        "category": "Osseous Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "osteoma",
            "skull osteoma",
            "cranial osteoma",
            "ossifying fibroma",
            "calvarial osteoma",
            "d16.4",
            "d164",
        ],
    },

    # ── Pituitary Tumors ──────────────────────────────────────────────────────

    "PITNET": {
        "icd10_code": "D35.2",
        "icd10_description": "Benign neoplasm of pituitary gland (Pituitary Neuroendocrine Tumor)",
        "category": "Pituitary Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "pitnet",
            "pituitary adenoma",
            "pituitary tumor",
            "pituitary neuroendocrine tumor",
            "pituitary net",
            "pit net",
            "adenoma",
            "microadenoma",
            "macroadenoma",
            "sellar tumor",
            "d35.2",
            "d352",
        ],
    },

    # ── Plasma Cell Neoplasms ─────────────────────────────────────────────────

    "PLASMOCYTOMA": {
        "icd10_code": "C90.30",
        "icd10_description": "Solitary plasmacytoma not having achieved remission",
        "category": "Plasma Cell Neoplasm",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "plasmocytoma",
            "plasmacytoma",
            "solitary plasmacytoma",
            "plasma cell tumor",
            "intracranial plasmacytoma",
            "c90.30",
            "c9030",
        ],
    },

    # ── Post-operative ────────────────────────────────────────────────────────

    "POSTOP_MRI": {
        "icd10_code": "Z48.89",
        "icd10_description": "Encounter for other specified surgical aftercare",
        "category": "Post-operative",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "postop",
            "post-op",
            "post op",
            "postoperative",
            "post-operative",
            "follow-up",
            "followup",
            "post surgery",
            "surgical aftercare",
            "postop mri",
            "z48.89",
            "z4889",
        ],
    },

    # ── Nerve Sheath Tumors ───────────────────────────────────────────────────

    "SCHWANNOMA": {
        "icd10_code": "D33.3",
        "icd10_description": "Benign neoplasm of cranial nerves (Schwannoma / Acoustic Neuroma)",
        "category": "Nerve Sheath Tumor",
        "dataset": ["Gazi 2025"],
        "aliases": [
            "schwannoma",
            "acoustic neuroma",
            "vestibular schwannoma",
            "neurinoma",
            "neurilemoma",
            "acoustic tumor",
            "cn viii tumor",
            "eighth nerve tumor",
            "d33.3",
            "d333",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Reverse-lookup indices (built once, on first use)
# ─────────────────────────────────────────────────────────────────────────────

_alias_to_folders: Optional[Dict[str, List[str]]] = None   # alias_lc → [folder, …]
_icd_to_folders:   Optional[Dict[str, List[str]]] = None   # icd_lc   → [folder, …]


def _build_indices() -> None:
    """Build reverse-lookup dictionaries from ICD10_MAPPING (idempotent)."""
    global _alias_to_folders, _icd_to_folders
    if _alias_to_folders is not None:
        return  # already built

    _alias_to_folders = {}
    _icd_to_folders   = {}

    for folder, info in ICD10_MAPPING.items():
        folder_lc = folder.lower()

        # 1. Folder name itself
        _alias_to_folders.setdefault(folder_lc, []).append(folder)

        # 2. ICD code — both canonical (with dot) and compact (without dot)
        icd_lc       = info["icd10_code"].lower()
        icd_nodot_lc = icd_lc.replace(".", "")
        _icd_to_folders.setdefault(icd_lc,       []).append(folder)
        _icd_to_folders.setdefault(icd_nodot_lc, []).append(folder)

        # 3. All declared clinical/abbreviation aliases
        for alias in info["aliases"]:
            a = alias.lower()
            _alias_to_folders.setdefault(a, []).append(folder)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resolve_query_to_folders(query: str) -> List[str]:
    """
    Resolve a free-text search query to a list of dataset folder names.

    Matching strategy (in order of priority):
      1. Exact alias match  (alias_to_folders)
      2. Exact ICD-10 code match  (icd_to_folders, with/without dot)
      3. Partial / substring match across all aliases and ICD codes

    Args:
        query: Raw search string (case-insensitive).

    Returns:
        Deduplicated, sorted list of matching uppercase folder names.
        Returns an empty list when no mapping is found (caller falls back
        to plain text search).
    """
    _build_indices()

    q = query.strip().lower()
    if not q:
        return []

    found: set[str] = set()

    # ── Priority 1 & 2 : exact matches ──────────────────────────────────────
    if q in _alias_to_folders:
        found.update(_alias_to_folders[q])
    if q in _icd_to_folders:
        found.update(_icd_to_folders[q])

    # ── Priority 3 : substring / partial match ───────────────────────────────
    if not found:
        for alias, folders in _alias_to_folders.items():
            if q in alias or alias in q:
                found.update(folders)
        for icd, folders in _icd_to_folders.items():
            if q in icd or icd in q:
                found.update(folders)

    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# Future-Ready: Abdomen CT Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Current status (March 2026):
#   • Local folder: segpro-med/abdomen/
#   • Structure: flat by patient-ID (20020, 20025, 20028, 20035)
#   • Labels: anatomical segmentation index only (e.g. "76: * Vertebra")
#   • No diagnostic class subfolders → mapping not active yet
#
# When diagnostic class subfolders are added, uncomment and adjust:
#
# "VERTEBRA_FRACTURE": {
#     "icd10_code": "S32.009A",
#     "icd10_description": "Unspecified fracture of unspecified lumbar vertebra, initial encounter",
#     "category": "Spinal Pathology",
#     "dataset": ["Abdomen CT"],
#     "aliases": [
#         "vertebra fracture", "lumbar fracture", "spinal fracture",
#         "compression fracture", "vertebral body fracture",
#         "s32.009a", "s32009a",
#     ],
# },
# "SPONDYLOSIS": {
#     "icd10_code": "M47.816",
#     "icd10_description": "Spondylosis with radiculopathy, lumbar region",
#     "category": "Spinal Pathology",
#     "dataset": ["Abdomen CT"],
#     "aliases": [
#         "spondylosis", "lumbar spondylosis", "degenerative disc",
#         "disc degeneration", "lumbar osteoarthritis",
#         "m47.816", "m47816",
#     ],
# },
# "LIVER_LESION": {
#     "icd10_code": "K76.89",
#     "icd10_description": "Other specified diseases of liver",
#     "category": "Hepatic Pathology",
#     "dataset": ["Abdomen CT"],
#     "aliases": [
#         "liver lesion", "hepatic lesion", "liver mass",
#         "focal liver lesion", "fll",
#         "k76.89", "k7689",
#     ],
# },
# "KIDNEY_CYST": {
#     "icd10_code": "N28.1",
#     "icd10_description": "Cyst of kidney, acquired",
#     "category": "Renal Pathology",
#     "dataset": ["Abdomen CT"],
#     "aliases": [
#         "kidney cyst", "renal cyst", "simple renal cyst",
#         "n28.1", "n281",
#     ],
# },
# "AAA": {
#     "icd10_code": "I71.4",
#     "icd10_description": "Abdominal aortic aneurysm, without rupture",
#     "category": "Vascular Pathology",
#     "dataset": ["Abdomen CT"],
#     "aliases": [
#         "aaa", "abdominal aortic aneurysm", "aortic aneurysm",
#         "i71.4", "i714",
#     ],
# },

# ─────────────────────────────────────────────────────────────────────────────
# Future-Ready: Mammography (MG) Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Current status (March 2026):
#   • Local folder: segpro-med/MG/836163459/
#   • Structure: single study UID folder with 4 view DICOMs
#                (LCC.dcm, LMLO.dcm, RCC.dcm, RMLO.dcm)
#   • No BI-RADS or diagnostic category subfolders → mapping not active yet
#   • VLM (MedGemma) analysis annotates per-case, not by folder class
#
# When BI-RADS / pathology category subfolders are added, uncomment:
#
# "BREAST_CANCER": {
#     "icd10_code": "C50.919",
#     "icd10_description": "Malignant neoplasm of unspecified site of unspecified female breast",
#     "category": "Breast Malignancy",
#     "dataset": ["MG"],
#     "aliases": [
#         "breast cancer", "breast carcinoma", "mammary carcinoma",
#         "birads 5", "birads5", "bi-rads 5",
#         "c50.919", "c50919",
#     ],
# },
# "DCIS": {
#     "icd10_code": "D05.10",
#     "icd10_description": "Intraductal carcinoma in situ of unspecified breast",
#     "category": "Breast Malignancy",
#     "dataset": ["MG"],
#     "aliases": [
#         "dcis", "ductal carcinoma in situ", "intraductal carcinoma",
#         "birads 4", "birads4",
#         "d05.10", "d0510",
#     ],
# },
# "BENIGN_MASS": {
#     "icd10_code": "N63.0",
#     "icd10_description": "Unspecified lump in unspecified breast",
#     "category": "Breast Benign",
#     "dataset": ["MG"],
#     "aliases": [
#         "benign mass", "benign lump", "fibroadenoma", "cyst",
#         "birads 3", "birads3",
#         "n63.0", "n630",
#     ],
# },
# "SCREENING_NORMAL": {
#     "icd10_code": "Z12.31",
#     "icd10_description": "Encounter for screening mammogram for malignant neoplasm of breast",
#     "category": "Breast Normal / Screening",
#     "dataset": ["MG"],
#     "aliases": [
#         "screening", "normal mammogram", "birads 1", "birads 2",
#         "negative", "benign findings",
#         "z12.31", "z1231",
#     ],
# },


def get_icd10_for_folder(folder: str) -> Optional[Dict]:
    """
    Return the full ICD-10 info dictionary for a given folder name.

    Args:
        folder: Folder name (case-insensitive, e.g. "cavernoma", "CAVERNOMA").

    Returns:
        The mapping dictionary, or None if not found.
    """
    return ICD10_MAPPING.get(folder.upper())


def get_folders_for_icd10(icd_code: str) -> List[str]:
    """
    Return all folder names that map to the given ICD-10 code.

    Args:
        icd_code: ICD-10 code string, e.g. "D18.02" or "d1802".

    Returns:
        List of uppercase folder names.
    """
    _build_indices()
    code     = icd_code.strip().lower()
    code_dot = code.replace(".", "")
    return sorted(set(
        _icd_to_folders.get(code, []) +
        _icd_to_folders.get(code_dot, [])
    ))


def list_all_mappings() -> List[Tuple[str, str, str, str, List[str]]]:
    """
    Return the full mapping table as a list of tuples, sorted by folder name.

    Returns:
        List of (folder, icd10_code, icd10_description, category, datasets) tuples.
    """
    return [
        (
            folder,
            info["icd10_code"],
            info["icd10_description"],
            info["category"],
            info["dataset"],
        )
        for folder, info in sorted(ICD10_MAPPING.items())
    ]


def get_category_summary() -> Dict[str, List[str]]:
    """
    Return a dict grouping folder names by clinical category.

    Returns:
        { category_name: [folder, …] }
    """
    summary: Dict[str, List[str]] = {}
    for folder, info in ICD10_MAPPING.items():
        summary.setdefault(info["category"], []).append(folder)
    return {cat: sorted(folders) for cat, folders in sorted(summary.items())}
