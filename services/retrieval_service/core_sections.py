"""
A small, hand-verified table of the actually-correct sections for common
topics — a higher-precision supplement to the auto-derived graph
relationships (which are mostly keyword co-occurrence "MENTIONS" edges, not
real relevance signals; see graph_store.py's docstring).

Every ID below was checked against its real content during testing, not
guessed — this table exists specifically because pure embedding similarity
and noisy auto-derived graph edges have repeatedly, concretely failed to
surface these exact sections for natural citizen phrasing (helmet, fine
amounts, police authority during a stop). These are added as ADDITIONAL
graph candidates — they still compete on real relevance score (see
routes.py/fusion.py), they are not force-included regardless of fit. This
keeps the "real scores decide" architecture intact; it just gives the
scorer a better, curated pool to choose from instead of relying only on
whatever the noisy MENTIONS edges happened to surface.

Extend this table only after verifying a section's actual content — the
same discipline used to build it, not by assumption.
"""

CORE_SECTIONS: dict[str, list[str]] = {
    "Helmet": ["HR_MVA_129", "HR_MVA_194D"],  # requirement; ₹1,000 penalty + 3-month disqualification
    "Fine": ["HR_MVA_177"],  # general default penalty when no specific one is set
    "Registration Certificate": ["HR_MVA_39", "HR_MVA_130", "HR_MVA_158", "HR_MVA_192"],
    "Driving Licence": ["HR_MVA_3", "HR_MVA_130", "HR_MVA_181", "HR_MVA_16"],
    "Challan": ["HR_MVA_200"],  # composition (compounding) of offences
    "Traffic Signal": ["HR_MVA_116", "HR_MVA_119"],
    "Seat Belt": ["HR_MVA_194B"],  # requirement + ₹1,000 penalty, combined in one section
    "Speed Limit": ["HR_MVA_112", "HR_MVA_183"],
    "Parking": ["HR_MVA_117"],  # designates where parking is permitted — lighter verification than the rest
    # Police Officer's bundle specifically covers "what can an officer do
    # during a roadside stop" — the exact gap that let the model over-extend
    # a real-but-mismatched premises-search clause onto a roadside question.
    "Police Officer": ["HR_MVA_130", "HR_MVA_206", "HR_MVA_207", "HR_MVA_202"],
}


def core_section_ids(matched_entities: list[str]) -> list[str]:
    ids: list[str] = []
    for entity in matched_entities:
        for record_id in CORE_SECTIONS.get(entity, []):
            if record_id not in ids:
                ids.append(record_id)
    return ids
