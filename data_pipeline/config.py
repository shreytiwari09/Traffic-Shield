"""
Data-pipeline configuration — paths, source registry, and the GraphRAG schema.

Self-contained: the pipeline is the offline database-building stage and depends
on no serving application, so it can be rebuilt against whatever app is put in
front of it.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "trafficshield_kb"

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PARSED_MD_DIR = DATA_DIR / "parsed_markdown"
CLEANED_DIR = DATA_DIR / "cleaned_sections"
GRAPH_DIR = DATA_DIR / "graph"

DATASET_PATH = DATA_DIR / "dataset.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
VALIDATION_REPORT_PATH = DATA_DIR / "validation_report.json"
REJECTED_PATH = DATA_DIR / "rejected_records.jsonl"
ENTITIES_PATH = GRAPH_DIR / "entities.json"
RELATIONSHIPS_PATH = GRAPH_DIR / "relationships.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

ALL_DIRS = [DATA_DIR, RAW_PDF_DIR, PARSED_MD_DIR, CLEANED_DIR, GRAPH_DIR]

# ---------------------------------------------------------------------------
# Source registry
#
# One entry per required source in Phase 1 of the blueprint. ``parser_style``
# selects the section-header grammar used in Phase 4 (see phase4_sections.py).
# ``priority`` follows the project brief: 1 = primary legislation … 7 = other.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "motor_vehicles_act_1988": {
        "act": "Motor Vehicles Act, 1988",
        "unit": "section",
        "jurisdiction": "India",
        "authority": "Government of India",
        "document_type": "Act",
        "state": "Haryana",
        "priority": 1,
        "parser_style": "act",
    },
    "motor_vehicles_amendment_2019": {
        "act": "Motor Vehicles (Amendment) Act, 2019",
        "unit": "section",
        "jurisdiction": "India",
        "authority": "Government of India",
        "document_type": "Act",
        "state": "Haryana",
        "priority": 1,
        "parser_style": "amendment",
    },
    "central_motor_vehicle_rules_1989": {
        "act": "Central Motor Vehicles Rules, 1989",
        "unit": "rule",
        "jurisdiction": "India",
        "authority": "Government of India",
        "document_type": "Rule",
        "state": "Haryana",
        "priority": 2,
        "parser_style": "rule_heading",
    },
    "motor_vehicles_driving_regulations_2017": {
        "act": "Motor Vehicles (Driving) Regulations, 2017",
        "unit": "regulation",
        "jurisdiction": "India",
        "authority": "Government of India",
        "document_type": "Regulation",
        "state": "Haryana",
        "priority": 3,
        "parser_style": "regulation",
    },
    "haryana_motor_vehicle_rules": {
        "act": "Haryana Motor Vehicle Rules, 1993",
        "unit": "rule",
        "jurisdiction": "Haryana",
        "authority": "Government of Haryana",
        "document_type": "Rule",
        "state": "Haryana",
        "priority": 2,
        "parser_style": "haryana",
    },
    # Not traffic law itself — a general central Act on evidence/procedure.
    # Included because it governs what counts as admissible evidence in a
    # traffic matter (e.g. e-challan records, dashcam/CCTV footage, digital
    # signatures on notices). Given its own priority tier (4) rather than
    # tier 1 alongside the Motor Vehicles Act, since it's only relevant to a
    # narrow slice of questions (admissibility), not traffic conduct itself.
    "bharatiya_sakshya_adhiniyam_2023": {
        "act": "Bharatiya Sakshya Adhiniyam, 2023",
        "unit": "section",
        "jurisdiction": "India",
        "authority": "Government of India",
        "document_type": "Act",
        "state": "Haryana",
        "priority": 4,
        "parser_style": "act",
    },
}

# Short prefix used to build record IDs (HR_MVA_119, HR_HMVR_11, …).
ID_PREFIXES: dict[str, str] = {
    "motor_vehicles_act_1988": "MVA",
    "motor_vehicles_amendment_2019": "MVA19",
    "central_motor_vehicle_rules_1989": "CMVR",
    "motor_vehicles_driving_regulations_2017": "MVDR",
    "haryana_motor_vehicle_rules": "HMVR",
    "bharatiya_sakshya_adhiniyam_2023": "BSA",
}

STATE = "Haryana"

DEFAULT_SOURCE_META = {
    "act": "Unknown Document",
    "unit": "section",
    "jurisdiction": "India",
    "authority": "Unknown",
    "document_type": "Other",
    "state": STATE,
    "priority": 7,
    "parser_style": "act",
}

# ---------------------------------------------------------------------------
# Phase 5 — required fields on every dataset record (blueprint §5)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ["id", "act", "section", "title", "content", "page", "source_pdf", "state"]

# A legal section shorter than this is almost certainly a mis-detected header
# (a stray cross-reference or a surviving TOC line) rather than real text.
# Kept low so genuinely brief provisions — "Short title and commencement" runs
# to about 70 characters — survive; the monotonic-numbering filter in Phase 4
# does the heavy lifting against false headers.
MIN_CONTENT_CHARS = 60

# ---------------------------------------------------------------------------
# Phase 6 — chunking (blueprint §6)
# ---------------------------------------------------------------------------
CHUNK_TARGET_TOKENS = 800      # upper bound of the 500–800 band
CHUNK_MIN_TOKENS = 500         # lower bound; sections below this stay whole
CHUNK_OVERLAP_RATIO = 0.15     # 15 %
CHROMA_COLLECTION = "trafficshield_sections"

# Embeddings are served locally by Ollama — no hosted API at any pipeline stage.
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Phase 7 — GraphRAG schema (blueprint §7)
# ---------------------------------------------------------------------------
# Canonical entity -> surface forms searched for in section text (lowercased,
# matched on word boundaries).
ENTITY_ALIASES: dict[str, list[str]] = {
    "Driver": ["driver", "drivers", "driving licence holder", "person driving"],
    "Police Officer": [
        "police officer",
        "police officers",
        "officer in uniform",
        "officer authorised",
        "authorised officer",
        "traffic police",
        "enforcement officer",
        # The enforcing authority under the Haryana rules is often an officer
        # of the transport department rather than the police.
        "officer of the Motor Vehicles Department",
        "Motor Vehicles Department",
    ],
    "Vehicle": ["motor vehicle", "motor vehicles", "vehicle", "vehicles", "transport vehicle"],
    "Driving Licence": ["driving licence", "driving license", "learner's licence", "licence to drive"],
    "Registration Certificate": [
        "certificate of registration",
        "registration certificate",
        "registration mark",
    ],
    # The statutes rarely print the word "challan" in its enforcement sense —
    # a challan is issued under the compounding machinery of MV Act s.200, and
    # most literal "challan" mentions in these documents are treasury challans
    # (fee-payment slips). The statutory vocabulary is listed alongside it so
    # the graph reflects the law rather than the colloquial term.
    "Challan": [
        "challan",
        "chalan",
        "e-challan",
        "compounding of offence",
        "compounding of offences",
        "compounded",
        "notice of offence",
        "composition of the offence",
    ],
    "Fine": ["fine", "fines", "penalty", "penalties", "punishable with fine"],
    "Traffic Signal": [
        "traffic signal",
        "traffic signals",
        "traffic light",
        "traffic sign",
        "road sign",
        "signage",
    ],
    "Helmet": ["helmet", "helmets", "protective headgear"],
    "Seat Belt": ["seat belt", "seat belts", "safety belt", "seatbelt"],
    "Parking": ["parking", "parked", "park a motor vehicle"],
    "Speed Limit": ["speed limit", "speed limits", "speeding", "limits of speed"],
    "Legal Section": [],  # instantiated per dataset record, not keyword-matched
}

ENTITY_TYPES: dict[str, str] = {
    "Driver": "Person",
    "Police Officer": "Authority",
    "Vehicle": "Object",
    "Driving Licence": "Document",
    "Registration Certificate": "Document",
    "Challan": "Instrument",
    "Fine": "Consequence",
    "Traffic Signal": "Control",
    "Helmet": "Equipment",
    "Seat Belt": "Equipment",
    "Parking": "Activity",
    "Speed Limit": "Constraint",
    "Legal Section": "Law",
}

# Fixed domain relationships from the blueprint. Each carries the alias pair
# that must BOTH appear in a section for the edge to be evidenced by it.
DOMAIN_RELATIONSHIPS: list[dict] = [
    {"source": "Driver", "relation": "MUST_OBEY", "target": "Traffic Signal"},
    {"source": "Police Officer", "relation": "MAY_ASK", "target": "Driving Licence"},
    {"source": "Police Officer", "relation": "MAY_ISSUE", "target": "Challan"},
    {"source": "Challan", "relation": "HAS_FINE", "target": "Fine"},
    {"source": "Legal Section", "relation": "GOVERNS", "target": "Driver"},
]
