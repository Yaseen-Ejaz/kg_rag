from neo4j import GraphDatabase
from openai import OpenAI
import re
import json
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


class GraphRAGSystem:
    def __init__(self, uri, user, password, openai_api_key, openai_model):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        if not openai_api_key:
            raise ValueError("Missing OpenAI API key.")

        # The OpenAI client you used previously
        self.openai = OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model
        
        # Caching for subgraph queries
        self._subgraph_cache = {}
        self._node_match_cache = {}

        print("\n[+] Connected to Neo4j")
        print("[+] OpenAI LLM ready")
        print("[+] Performance optimizations enabled\n")

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
    @lru_cache(maxsize=1)
    def get_schema(self):
        """Cached schema extraction - only runs once"""
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
        response = self._call_llm(prompt, max_tokens=100)  # Reduced from 200

        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                
                # Limit number of entities to prevent explosion
                if len(entities) > 5:
                    print(f"[!] Truncating entities from {len(entities)} to 5 for performance")
                    entities = entities[:5]
                
                print(f"\n[+] Extracted entities: {entities}")
                return entities
            return []
        except Exception as e:
            print(f"[!] Entity extraction error: {e}")
            return []

    # ================= Node matching (fuzzy) with caching =================
    def find_matching_nodes(self, entity_name, limit=5):
        """Return list of candidate nodes for an entity name (fuzzy match) - cached."""
        cache_key = (entity_name, limit)
        
        if cache_key in self._node_match_cache:
            print(f"[CACHE HIT] Using cached node matches for '{entity_name}'")
            return self._node_match_cache[cache_key]
        
        query_exact = """
        MATCH (n)
        WHERE n.name = $name
        RETURN n.name AS name, labels(n) AS labels, elementId(n) AS id
        LIMIT $limit
        """

        res = self.run_cypher(query_exact, {"name": entity_name, "limit": limit})
        if res:
            self._node_match_cache[cache_key] = res
            return res

        query_fuzzy = """
        MATCH (n)
        WHERE toLower(coalesce(n.name,'')) CONTAINS toLower($name)
        RETURN n.name AS name, labels(n) AS labels, elementId(n) AS id
        LIMIT $limit
        """
        res = self.run_cypher(query_fuzzy, {"name": entity_name, "limit": limit})
        print(f"\n[+] Found {len(res)} matching nodes for '{entity_name}'")
        
        self._node_match_cache[cache_key] = res
        return res

    # ================= APOC-based multi-hop traversal with caching =================
    def apoc_subgraph(self, start_node_name, max_depth=5, max_nodes=100):  # Reduced from 300
        """Use APOC to get the subgraph around a starting node up to max_depth - cached.

        Returns a dict with nodes (list of maps) and relationships (list of maps).
        """
        # Cache key based on parameters
        cache_key = (start_node_name, max_depth, max_nodes)
        
        if cache_key in self._subgraph_cache:
            print(f"[CACHE HIT] Using cached subgraph for {start_node_name}")
            return self._subgraph_cache[cache_key]
        
        # Find a start node fuzzily and run apoc.path.subgraphAll
        query = """
        MATCH (start)
        WHERE toLower(coalesce(start.name, '')) CONTAINS toLower($name)
        WITH start
        CALL apoc.path.subgraphAll(start, {maxLevel:$max_depth, limit:$max_nodes}) YIELD nodes, relationships
        RETURN [n IN nodes | {id:id(n), name:coalesce(n.name, toString(id(n))), labels:labels(n), props:apoc.map.fromPairs([k IN keys(n) | [k, n[k]]])}] AS nodes,
               [r IN relationships | {id:id(r), type:type(r), start:coalesce(startNode(r).name, toString(id(startNode(r)))), end:coalesce(endNode(r).name, toString(id(endNode(r))))}] AS relationships
        LIMIT 1
        """

        records = self.run_cypher(query, {"name": start_node_name, "max_depth": max_depth, "max_nodes": max_nodes})

        if not records:
            result = {"start_node": start_node_name, "nodes": [], "relationships": [], "total_nodes": 0, "total_edges": 0}
            self._subgraph_cache[cache_key] = result
            return result

        rec = records[0]
        nodes = rec.get("nodes", [])
        rels = rec.get("relationships", [])

        # Optionally trim nodes/relationships to max_nodes
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
        if len(rels) > max_nodes * 2:
            rels = rels[: max_nodes * 2]

        result = {
            "start_node": start_node_name,
            "nodes": {n["name"]: {"id": n["id"], "labels": n["labels"], "props": n["props"]} for n in nodes},
            "edges": [{"source": r["start"], "target": r["end"], "relationship": r["type"]} for r in rels],
            "total_nodes": len(nodes),
            "total_edges": len(rels)
        }
        
        # Cache the result
        self._subgraph_cache[cache_key] = result
        return result

    # ================= Assemble context using APOC-subgraph with parallel processing =================
    def assemble_bfs_context(self, entities, max_depth=5, max_nodes=100):  # Reduced defaults
        """Assemble context with parallel processing for better performance."""
        context = []
        
        # Collect all tasks first
        tasks = []
        for entity in entities:
            if entity.lower() in ['movie', 'person', 'company', 'film', 'actor', 'director', 'thing', 'someone']:
                print(f"[*] Skipping generic entity: {entity}")
                continue

            matches = self.find_matching_nodes(entity, limit=3)
            for match in matches[:2]:  # Only process top 2 matches per entity
                tasks.append((entity, match['name']))
        
        if not tasks:
            return context
        
        # Process tasks in parallel
        print(f"[PARALLEL] Processing {len(tasks)} subgraph queries with {min(3, len(tasks))} workers")
        with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as executor:
            future_to_task = {
                executor.submit(self.apoc_subgraph, node_name, max_depth, max_nodes): (entity, node_name)
                for entity, node_name in tasks
            }
            
            for future in as_completed(future_to_task):
                entity, node_name = future_to_task[future]
                try:
                    subgraph = future.result()
                    if subgraph and subgraph['total_nodes'] > 0:
                        context.append({
                            "entity": entity,
                            "matched_node": node_name,
                            "subgraph": subgraph
                        })
                except Exception as e:
                    print(f"[!] Error processing subgraph for {node_name}: {e}")
        
        return context

    # ================= Format subgraph for LLM =================
    def format_subgraph_for_llm(self, subgraph, max_edges=20):  # Reduced from 40
        """Format subgraph with reduced size for faster LLM processing."""
        text = f"Starting from: {subgraph['start_node']}\n\n"
        text += "NODES:\n"
        
        # Only show first 50 nodes instead of 200
        for node_name, node_info in list(subgraph['nodes'].items())[:50]:
            labels = ", ".join(node_info.get('labels', []))
            text += f"  - {node_name} (Type: {labels})\n"

        if len(subgraph['nodes']) > 50:
            text += f"  ... and {len(subgraph['nodes']) - 50} more nodes\n"

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
            context_text += self.format_subgraph_for_llm(ctx['subgraph'], max_edges=30)
            context_text += "\n"

        if len(context_text) > 10000:  # Reduced from 12000
            context_text = context_text[:10000] + "\n\n... (context truncated for length)"
            print("[!] Context truncated to fit token limit")

        prompt = f"""
            You are an expert in the domain of knowledge graphs. You are answering questions using a knowledge graph that was explored via APOC subgraph traversal.

            {context_text}

            QUESTION: {question}

            INSTRUCTIONS:
            - Use ONLY the nodes and relationships shown above
            - Follow the relationship paths to answer the question
            - Be explicit about which nodes/relationships led to the answer (put a tick in the start if found in the graph: ✅)
            - If the subgraph doesn't contain the answer, say so (put a cross in the start if not found in the graph: ❌)
            - Keep answer concise (2-4 sentences)
            - Give the path in the last line in the format with a blank line before it: Path: Node1 --[RELATIONSHIP]--> Node2 --[RELATIONSHIP]--> ...

            ANSWER:
            """
        answer = self._call_llm(prompt, max_tokens=400)
        return answer

    # ================= Main pipeline with detailed timing =================
    def ask(self, question, max_depth=5, max_nodes_per_entity=100):  # Reduced default
        # Start timing
        start_time = time.time()
        timings = {}
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")

        # Entity extraction
        entity_start = time.time()
        entities = self.extract_entities(question)
        timings['entity_extraction'] = time.time() - entity_start
        
        if not entities:
            print("[!] No entities found, trying a schema-based fallback")
            elapsed_time = time.time() - start_time
            return {
                "question": question,
                "entities": [],
                "context": [],
                "answer": "No specific entities found in question. Please rephrase with specific names or concepts.",
                "processing_time": round(elapsed_time, 2),
                "timing_breakdown": {k: round(v, 2) for k, v in timings.items()}
            }

        # Context assembly
        print("\n[STEP] Building context via APOC subgraph traversal...")
        context_start = time.time()
        context = self.assemble_bfs_context(entities, max_depth=max_depth, max_nodes=max_nodes_per_entity)
        timings['context_assembly'] = time.time() - context_start

        if not context:
            elapsed_time = time.time() - start_time
            timings['answer_generation'] = 0
            return {
                "question": question,
                "entities": entities,
                "context": [],
                "answer": f"Could not find nodes matching: {entities}",
                "processing_time": round(elapsed_time, 2),
                "timing_breakdown": {k: round(v, 2) for k, v in timings.items()}
            }

        # Answer generation
        answer_start = time.time()
        answer = self.generate_answer_from_bfs(question, context)
        timings['answer_generation'] = time.time() - answer_start
        
        # End timing
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"[TIMING BREAKDOWN]")
        print(f"  Entity extraction: {timings['entity_extraction']:.2f}s")
        print(f"  Context assembly: {timings['context_assembly']:.2f}s")
        print(f"  Answer generation: {timings['answer_generation']:.2f}s")
        print(f"  Total: {elapsed_time:.2f}s")
        print(f"{'='*60}\n")

        return {
            "question": question,
            "entities": entities,
            "context": context,
            "answer": answer,
            "stats": {
                "subgraphs_explored": len(context),
                "total_nodes": sum(ctx["subgraph"]["total_nodes"] for ctx in context),
                "total_edges": sum(ctx["subgraph"]["total_edges"] for ctx in context)
            },
            "processing_time": round(elapsed_time, 2),
            "timing_breakdown": {k: round(v, 2) for k, v in timings.items()}
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
        
        if "processing_time" in result:
            print(f"Processing Time: {result['processing_time']}s")
            
        if "timing_breakdown" in result:
            print(f"\nTiming Breakdown:")
            for phase, duration in result['timing_breakdown'].items():
                print(f"  {phase}: {duration}s")

        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")
    
    # ================= Cache management =================
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._subgraph_cache.clear()
        self._node_match_cache.clear()
        print("[+] All caches cleared")
    
    def get_cache_stats(self):
        """Get statistics about cache usage."""
        return {
            "subgraph_cache_size": len(self._subgraph_cache),
            "node_match_cache_size": len(self._node_match_cache)
        }

    def close(self):
        self.driver.close()