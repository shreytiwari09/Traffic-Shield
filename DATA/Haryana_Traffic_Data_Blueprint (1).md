# Data Blueprint --- Haryana Traffic Legal Assistant

> Scope: This document covers **only the data pipeline** (collection,
> cleaning, storage, vectorization, and GraphRAG). It does not include
> app or model development.

## Phase 1 --- Data Collection

### Objective

Collect only official legal documents related to traffic laws in
Haryana.

### Required Sources

-   Motor Vehicles Act, 1988
-   Central Motor Vehicle Rules
-   Haryana Motor Vehicle Rules
-   Haryana traffic challan notifications
-   Driver rights and traffic enforcement notifications

### Output

``` text
data/raw_pdfs/
```

Store every PDF with its original filename.

------------------------------------------------------------------------

## Phase 2 --- PDF Parsing

### Objective

Convert PDFs into structured text while preserving legal formatting.

### Process

-   Convert PDF → Markdown
-   Preserve headings
-   Preserve section numbers
-   Preserve tables and lists
-   Keep original page references

### Output

``` text
data/parsed_markdown/
```

------------------------------------------------------------------------

## Phase 3 --- Data Cleaning

### Remove Only

-   Table of contents
-   Index pages
-   Blank pages
-   Duplicate pages
-   Advertisement or non-legal pages

### Never Remove

-   Legal sections
-   Notes attached to sections
-   Tables containing penalties
-   Government notifications

### Output

``` text
data/cleaned_sections/
```

------------------------------------------------------------------------

## Phase 4 --- Legal Section Extraction

### Rule

Split documents by **legal section**, never by character count.

Each section becomes one record.

### Dataset Schema

  Field        Description
  ------------ -----------------------
  id           Unique record ID
  act          Name of Act/Rule
  section      Legal section number
  title        Section title
  content      Original legal text
  page         PDF page number
  source_pdf   Original PDF filename
  state        Haryana

### Example

``` json
{
  "id": "HR_119",
  "act": "Motor Vehicles Act",
  "section": "119",
  "title": "Duty of Driver",
  "content": "...original legal text...",
  "page": 142,
  "source_pdf": "motor_vehicles_act.pdf",
  "state": "Haryana"
}
```

### Output

``` text
dataset.jsonl
```

------------------------------------------------------------------------

## Phase 5 --- Metadata Validation

Every record must contain:

-   id
-   act
-   section
-   title
-   content
-   page
-   source_pdf
-   state

Reject incomplete records.

------------------------------------------------------------------------

## Phase 6 --- Vectorization

### Chunking Rules

-   Split only at section boundaries
-   Chunk size: 500--800 tokens
-   Overlap: 15%
-   Keep metadata attached to every chunk

### Output Structure

``` json
{
  "text": "...chunk...",
  "embedding": [],
  "metadata": {
    "section": "119",
    "page": 142,
    "act": "Motor Vehicles Act",
    "state": "Haryana"
  }
}
```

Store inside the vector database.

------------------------------------------------------------------------

## Phase 7 --- GraphRAG Data

### Entities

-   Driver
-   Police Officer
-   Vehicle
-   Driving Licence
-   Registration Certificate
-   Challan
-   Fine
-   Traffic Signal
-   Helmet
-   Seat Belt
-   Parking
-   Speed Limit
-   Legal Section

### Relationships

  Entity A         Relation    Entity B
  ---------------- ----------- -----------------
  Driver           MUST_OBEY   Traffic Signal
  Police Officer   MAY_ASK     Driving Licence
  Police Officer   MAY_ISSUE   Challan
  Challan          HAS_FINE    Fine
  Legal Section    GOVERNS     Driver

### Graph Output

``` text
graph/
├── entities.json
└── relationships.json
```

------------------------------------------------------------------------

# Final Deliverables

``` text
data/
├── raw_pdfs/
├── parsed_markdown/
├── cleaned_sections/
├── dataset.jsonl
└── graph/
    ├── entities.json
    └── relationships.json
```

This is the complete database-building stage before retrieval and
inference.
