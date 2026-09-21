// ===========================================================================
// Traffic Shield — Neo4j demo queries
//
// Paste into Neo4j Browser at http://localhost:7474  (neo4j / trafficshield)
// Run them IN ORDER: each one builds on the story the previous one told.
//
// Every query below RETURNS nodes/paths rather than scalars, because Browser
// only draws a graph when it receives graph objects — return count(*) and you
// get a table instead of a picture.
//
// Do NOT run `MATCH (n) RETURN n` — 1,331 nodes will hang the browser.
// ===========================================================================


// ---------------------------------------------------------------------------
// 1. THE DOMAIN MODEL — start here
//     5 hand-authored relationships. Tiny and completely readable: this is the
//     legal world modelled explicitly, not derived by a script.
// ---------------------------------------------------------------------------
MATCH path = ()-[:RELATES {kind: 'domain'}]->()
RETURN path;


// ---------------------------------------------------------------------------
// 2. THE MULTI-ENTITY GRAPH  ★ the one to show an evaluator
//     Legal sections that are evidence for FIVE OR MORE different concepts.
//     37 nodes, 128 edges — dense enough to look like a real graph, small
//     enough to read.
//
//     What it proves: sections are not leaves hanging off one concept, they
//     are BRIDGES between concepts. CMVR Rule 138 alone connects seven of
//     them (Driver, Police Officer, Vehicle, Driving Licence, Registration
//     Certificate, Helmet, Seat Belt). That shared structure is exactly what
//     a table cannot express and what vector search cannot traverse.
// ---------------------------------------------------------------------------
MATCH (s:Section)<-[:EVIDENCED_BY]-(e:Entity)
WHERE e.kind = 'concept'
WITH s, count(DISTINCT e) AS shared_by
WHERE shared_by >= 5
MATCH path = (s)<-[:EVIDENCED_BY]-(c:Entity)
WHERE c.kind = 'concept'
RETURN path;


// ---------------------------------------------------------------------------
// 3. ONE CONCEPT'S NEIGHBOURHOOD — the multi-hop story
//     Helmet -> the sections that evidence it -> the OTHER concepts those same
//     sections also touch. This is the two-hop walk a flat JSON lookup
//     structurally cannot make.
// ---------------------------------------------------------------------------
MATCH path = (h:Entity {name: 'Helmet'})-[:EVIDENCED_BY]->(s:Section)<-[:EVIDENCED_BY]-(other:Entity)
WHERE other.kind = 'concept' AND other.name <> 'Helmet'
RETURN path;


// ---------------------------------------------------------------------------
// 4. WHAT THE APP ACTUALLY RUNS — alias matching, live
//     This is retrieval_service/neo4j_store.py's real query. The question text
//     is matched against every concept's stored alias regexes inside the
//     database, then walked out to the evidencing sections.
//
//     Change the question string and re-run to show retrieval responding.
// ---------------------------------------------------------------------------
WITH 'what is the fine for not wearing a helmet?' AS question
MATCH (e:Entity)
WHERE any(p IN e.alias_patterns WHERE question =~ p)
MATCH path = (e)-[:EVIDENCED_BY]->(s:Section)
RETURN path;


// ---------------------------------------------------------------------------
// 5. SHARED LAW BETWEEN TWO CONCEPTS
//     Which sections govern BOTH a police officer and a driving licence?
//     A single hop in Cypher; a self-join with a subquery in SQL.
// ---------------------------------------------------------------------------
MATCH path = (a:Entity {name: 'Police Officer'})-[:EVIDENCED_BY]->(s:Section)<-[:EVIDENCED_BY]-(b:Entity {name: 'Driving Licence'})
RETURN path;


// ---------------------------------------------------------------------------
// 6. THE SAME ANSWER AS A TABLE — for the write-up, not the picture
//     Ranked exactly the way retrieval ranks it: by the source Act's legal
//     priority (1 = primary legislation ... 4 = supporting law).
// ---------------------------------------------------------------------------
MATCH (s:Section)<-[:EVIDENCED_BY]-(e:Entity)
WHERE e.kind = 'concept'
WITH s, collect(DISTINCT e.name) AS concepts, count(DISTINCT e) AS shared_by
WHERE shared_by >= 4
RETURN s.act AS act, s.section AS section, s.title AS title,
       shared_by, concepts, s.priority AS legal_priority
ORDER BY shared_by DESC, legal_priority, section
LIMIT 25;
