from neo4j import GraphDatabase
from openai import OpenAI
import re
import json
from collections import deque


class GraphRAGSystem:
    def __init__(self, uri, user, password, openai_api_key, openai_model="gpt-4o"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        if not openai_api_key:
            raise ValueError("Missing OpenAI API key.")

        # The OpenAI client you used previously
        self.openai = OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model

        print("\n[+] Connected to Neo4j")
        print("[+] OpenAI LLM ready\n")

    # ================= LLM helper =================
    def _call_llm(self, prompt, max_tokens=500):
        print("\n=== LLM CALL PROMPT ===")
        print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))

        resp = self.openai.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens
        )

        content = resp.choices[0].message.content
        print("\n=== LLM RAW OUTPUT ===")
        print(content)

        return content

    # ================= Neo4j helper =================
    def run_cypher(self, query, params=None):
        with self.driver.session() as session:
            return session.run(query, params or {}).data()

    # ================= Schema extraction (optional) =================
    def get_schema(self):
        schema = {}
        with self.driver.session() as session:
            nodes = session.run("CALL db.labels()").data()
            rels = session.run("CALL db.relationshipTypes()").data()
            prop_keys = session.run("CALL db.propertyKeys()").data()

        schema["labels"] = [n["label"] for n in nodes]
        schema["relationships"] = [r["relationshipType"] for r in rels]
        schema["properties"] = [p["propertyKey"] for p in prop_keys]

        return schema

    def build_schema_prompt(self):
        s = self.get_schema()
        return f"""
NODE LABELS: {s["labels"]}
RELATIONSHIP TYPES: {s["relationships"]}
PROPERTIES: {s["properties"]}
"""

    # ================= Entity extraction =================
    def extract_entities(self, question):
        """Use the LLM to extract potential entity names mentioned in question."""
        prompt = f"""
            Extract entity names from the question that might exist in a knowledge graph.

            GRAPH SCHEMA:
            {self.build_schema_prompt()}

            QUESTION: {question}

            Return ONLY a JSON list of potential entity names (proper nouns, names, organizations, etc).
            Format: ["Entity1", "Entity2"]
            If no entities found, return: []

            IMPORTANT: Return ONLY the JSON array, no other text.
            """
        response = self._call_llm(prompt, max_tokens=200)

        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                print(f"\n[+] Extracted entities: {entities}")
                return entities
            return []
        except Exception as e:
            print(f"[!] Entity extraction error: {e}")
            return []

    # ================= Node matching (fuzzy) =================
    def find_matching_nodes(self, entity_name, limit=5):
        """Return list of candidate nodes for an entity name (fuzzy match)."""
        query_exact = """
        MATCH (n)
        WHERE n.name = $name
        RETURN n.name AS name, labels(n) AS labels, elementId(n) AS id
        LIMIT $limit
        """

        res = self.run_cypher(query_exact, {"name": entity_name, "limit": limit})
        if res:
            return res

        query_fuzzy = """
        MATCH (n)
        WHERE toLower(coalesce(n.name,'')) CONTAINS toLower($name)
        RETURN n.name AS name, labels(n) AS labels, elementId(n) AS id
        LIMIT $limit
        """
        res = self.run_cypher(query_fuzzy, {"name": entity_name, "limit": limit})
        print(f"\n[+] Found {len(res)} matching nodes for '{entity_name}'")
        return res

    # ================= APOC-based multi-hop traversal =================
    def apoc_subgraph(self, start_node_name, max_depth=5, max_nodes=300):
        """Use APOC to get the subgraph around a starting node up to max_depth.

        Returns a dict with nodes (list of maps) and relationships (list of maps).
        """
        # Find a start node fuzzily and run apoc.path.subgraphAll
        query = """
        MATCH (start)
        WHERE toLower(coalesce(start.name, '')) CONTAINS toLower($name)
        WITH start
        CALL apoc.path.subgraphAll(start, {maxLevel:$max_depth}) YIELD nodes, relationships
        RETURN [n IN nodes | {id:id(n), name:coalesce(n.name, toString(id(n))), labels:labels(n), props:apoc.map.fromPairs([k IN keys(n) | [k, n[k]]])}] AS nodes,
               [r IN relationships | {id:id(r), type:type(r), start:coalesce(startNode(r).name, toString(id(startNode(r)))), end:coalesce(endNode(r).name, toString(id(endNode(r))))}] AS relationships
        LIMIT 1
        """

        records = self.run_cypher(query, {"name": start_node_name, "max_depth": max_depth})

        if not records:
            return {"start_node": start_node_name, "nodes": [], "relationships": [], "total_nodes": 0, "total_edges": 0}

        rec = records[0]
        nodes = rec.get("nodes", [])
        rels = rec.get("relationships", [])

        # Optionally trim nodes/relationships to max_nodes
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
        if len(rels) > max_nodes * 2:
            rels = rels[: max_nodes * 2]

        return {
            "start_node": start_node_name,
            "nodes": {n["name"]: {"id": n["id"], "labels": n["labels"], "props": n["props"]} for n in nodes},
            "edges": [{"source": r["start"], "target": r["end"], "relationship": r["type"]} for r in rels],
            "total_nodes": len(nodes),
            "total_edges": len(rels)
        }

    # ================= Assemble context using APOC-subgraph =================
    def assemble_bfs_context(self, entities, max_depth=7, max_nodes=300):
        context = []
        for entity in entities:
            if entity.lower() in ['movie', 'person', 'company', 'film', 'actor', 'director', 'thing', 'someone']:
                print(f"[*] Skipping generic entity: {entity}")
                continue

            matches = self.find_matching_nodes(entity, limit=3)
            for match in matches[:2]:
                node_name = match['name']
                subgraph = self.apoc_subgraph(node_name, max_depth=max_depth, max_nodes=max_nodes)
                if subgraph and subgraph['total_nodes'] > 0:
                    context.append({
                        "entity": entity,
                        "matched_node": node_name,
                        "subgraph": subgraph
                    })
        return context

    # ================= Format subgraph for LLM =================
    def format_subgraph_for_llm(self, subgraph, max_edges=40):
        text = f"Starting from: {subgraph['start_node']}\n\n"
        text += "NODES:\n"
        for node_name, node_info in list(subgraph['nodes'].items())[:200]:
            labels = ", ".join(node_info.get('labels', []))
            text += f"  - {node_name} (Type: {labels})\n"

        if len(subgraph['nodes']) > 200:
            text += f"  ... and {len(subgraph['nodes']) - 200} more nodes\n"

        text += "\nRELATIONSHIPS:\n"
        for edge in subgraph['edges'][:max_edges]:
            text += f"  - {edge['source']} --[{edge['relationship']}]--> {edge['target']}\n"

        if len(subgraph['edges']) > max_edges:
            text += f"  ... and {len(subgraph['edges']) - max_edges} more relationships\n"

        return text

    # ================= Generate answer using LLM =================
    def generate_answer_from_bfs(self, question, context):
        if not context:
            return "I couldn't find relevant information in the knowledge graph to answer your question."

        context_text = "KNOWLEDGE GRAPH (APOC SUBGRAPH):\n\n"
        for idx, ctx in enumerate(context, 1):
            context_text += f"=== Subgraph {idx}: {ctx['matched_node']} ===\n"
            context_text += self.format_subgraph_for_llm(ctx['subgraph'], max_edges=50)
            context_text += "\n"

        if len(context_text) > 12000:
            context_text = context_text[:12000] + "\n\n... (context truncated for length)"
            print("[!] Context truncated to fit token limit")

        prompt = f"""
            You are an expert in the domain of movies. You are answering questions using a knowledge graph that was explored via APOC subgraph traversal.

            {context_text}

            QUESTION: {question}

            INSTRUCTIONS:
            - Use ONLY the nodes and relationships shown above
            - Follow the relationship paths to answer the question
            - Be explicit about which nodes/relationships led to the answer
            - If the subgraph doesn't contain the answer, say so
            - Keep answer concise (2-4 sentences)

            ANSWER:
            """
        answer = self._call_llm(prompt, max_tokens=400)
        return answer

    # ================= Main pipeline =================
    def ask(self, question, max_depth=5, max_nodes_per_entity=300):
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")

        entities = self.extract_entities(question)
        if not entities:
            print("[!] No entities found, trying a schema-based fallback")
            return {
                "question": question,
                "entities": [],
                "context": [],
                "answer": "No specific entities found in question. Please rephrase with specific names or concepts."
            }

        print("\n[STEP] Building context via APOC subgraph traversal...")
        context = self.assemble_bfs_context(entities, max_depth=max_depth, max_nodes=max_nodes_per_entity)

        if not context:
            return {
                "question": question,
                "entities": entities,
                "context": [],
                "answer": f"Could not find nodes matching: {entities}"
            }

        answer = self.generate_answer_from_bfs(question, context)

        return {
            "question": question,
            "entities": entities,
            "context": context,
            "answer": answer,
            "stats": {
                "subgraphs_explored": len(context),
                "total_nodes": sum(ctx["subgraph"]["total_nodes"] for ctx in context),
                "total_edges": sum(ctx["subgraph"]["total_edges"] for ctx in context)
            }
        }

    # ================= Visualization helper =================
    def visualize_bfs_result(self, result):
        print("\n" + "="*60)
        print("BFS EXPLORATION SUMMARY")
        print("="*60)

        if "stats" in result:
            stats = result["stats"]
            print(f"Entities Found: {result['entities']}")
            print(f"Subgraphs Explored: {stats['subgraphs_explored']}")
            print(f"Total Nodes: {stats['total_nodes']}")
            print(f"Total Edges: {stats['total_edges']}")

        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")

    def close(self):
        self.driver.close()

