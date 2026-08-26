# Haryana Traffic Legal Assistant --- Data Planning

## Objective

Build a real-time voice legal assistant that helps drivers during
traffic police interactions in Haryana using only official legal
documents.

## Data Scope (Only Required Data)

### 1. Motor Vehicles Act

-   Driver rights
-   Driver duties
-   Traffic violations
-   Penalties & fines
-   Relevant legal sections

### 2. Haryana Traffic Rules

-   State-specific traffic regulations
-   Challan rules
-   Vehicle document requirements
-   Helmet, seatbelt, parking, speed rules

### 3. Police Authority

-   What a traffic officer can do
-   What a traffic officer cannot do
-   Documents an officer may legally ask for
-   Powers related to challan and vehicle seizure

## Data Structure

  Field        Description
  ------------ ------------------------------------
  act          Motor Vehicles Act / Haryana Rules
  section      Legal section number
  title        Section title
  content      Original legal text
  page         Source PDF page
  source_pdf   PDF filename

## Folder Structure

``` text
data/
├── raw_pdfs/
├── parsed_markdown/
├── cleaned_sections/
└── metadata/
```

## Processing Pipeline

1.  Collect official Haryana traffic PDFs.
2.  Convert PDFs → Markdown.
3.  Remove TOC, index, blank pages.
4.  Split by legal sections (not fixed characters).
5.  Store metadata with page and source.
6.  Create embeddings + GraphRAG index.

## Synthetic Data Policy

Do not generate synthetic laws.

Only generate conversation examples **after** storing the real legal
text.

Example: - Police: "Show your driving licence." - Driver: "Is it legally
required?" - Assistant: Answer using the actual legal section.
