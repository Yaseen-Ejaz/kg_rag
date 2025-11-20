from neo4j import GraphDatabase
from openai import OpenAI
import re
import json


class GraphRAGSystem:
    def __init__(self, uri, user, password, openai_api_key, openai_model="gpt-4o"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key.")
        
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

    # ================= Schema extraction =================
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

    # ================= Generate Cypher Query =================
    def generate_cypher_query(self, question, entities):
        """
        Generate a specific Cypher query for the question.
        This ensures we get EXACTLY the hops needed.
        """
        schema = self.build_schema_prompt()
        
        prompt = f"""
You are a Cypher query expert for Neo4j. Generate a Cypher query to answer the question.

GRAPH SCHEMA:
{schema}

ENTITIES FOUND: {entities}

QUESTION: {question}

RULES:
1. Generate ONE complete Cypher query
2. Use MATCH patterns that explicitly define each hop
3. For "actors who worked with actors", you need: Movie1 <- Actor1 -> Movie2 <- Actor2
4. Always return the final answer nodes with their properties
5. Use DISTINCT to avoid duplicates
6. Add LIMIT 50 at the end
7. Use proper relationship directions based on the schema

EXAMPLES:
Question: "Who directed Inception?"
Query: MATCH (m:Movie {{name: "Inception"}})-[r:DIRECTED_BY]->(d:Director) RETURN DISTINCT d.name

Question: "What movies did the director of Inception direct?"
Query: MATCH (m1:Movie {{name: "Inception"}})-[:DIRECTED_BY]->(d)-[:DIRECTED_BY]-(m2:Movie) 
       RETURN DISTINCT m2.name

Question: "Which actors worked with actors from Inception?"
Query: MATCH (m1:Movie {{name: "Inception"}})<-[:STARRED_ACTORS]-(a1:Actor)
       -[:STARRED_ACTORS]->(m2:Movie)<-[:STARRED_ACTORS]-(a2:Actor)
       WHERE a1 <> a2 AND m1 <> m2
       RETURN DISTINCT a2.name LIMIT 50

Return ONLY the Cypher query, no explanation. Use the actual entity names from: {entities}
"""
        
        response = self._call_llm(prompt, max_tokens=300)
        
        # Extract Cypher query
        # Look for text between ```cypher and ``` or just the query itself
        cypher_match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if cypher_match:
            query = cypher_match.group(1).strip()
        else:
            # If no code block, take the entire response
            query = response.strip()
        
        print(f"\n[+] Generated Cypher Query:\n{query}\n")
        return query

    # ================= Execute generated query =================
    def execute_generated_query(self, cypher_query):
        """Execute the generated Cypher query and return results."""
        try:
            results = self.run_cypher(cypher_query)
            print(f"[+] Query returned {len(results)} results")
            return results
        except Exception as e:
            print(f"[!] Query execution error: {e}")
            return None

    # ================= Entity extraction =================
    def extract_entities(self, question):
        """Extract entity names from question."""
        prompt = f"""
Extract entity names from the question that might exist in a knowledge graph.

GRAPH SCHEMA:
{self.build_schema_prompt()}

QUESTION: {question}

Return ONLY a JSON list of potential entity names (movie titles, actor names, director names, etc).
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

    # ================= Analyze question complexity =================
    def analyze_question_complexity(self, question):
        """Estimate required reasoning hops."""
        prompt = f"""
Analyze this knowledge graph query and determine how many relationship hops are needed.

QUESTION: {question}

Count the logical steps needed:
- "Who directed X?" = 1 hop (X -> director)
- "What movies did the director of X direct?" = 2 hops (X -> director -> movies)
- "Who acted in movies directed by director of X?" = 3 hops (X -> director -> movies -> actors)
- "Which actors worked with actors from movies by director of X?" = 4 hops (X -> director -> movies -> actors -> movies -> actors)
- "Which actors worked with actors who appeared in movies directed by director of X?" = 4 hops

Return ONLY a JSON object:
{{"estimated_hops": <number>, "reasoning": "<brief explanation>"}}
"""
        response = self._call_llm(prompt, max_tokens=150)
        
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                analysis = json.loads(match.group())
                estimated_hops = analysis.get("estimated_hops", 3)
                print(f"\n[+] Estimated hops needed: {estimated_hops}")
                print(f"[+] Reasoning: {analysis.get('reasoning', 'N/A')}")
                return estimated_hops
            return 3
        except Exception as e:
            print(f"[!] Complexity analysis error: {e}")
            return 3

    # ================= Format query results =================
    def format_query_results(self, results, max_results=100):
        """Format Cypher query results for LLM."""
        if not results:
            return "No results found from query."
        
        text = f"QUERY RESULTS ({len(results)} total):\n\n"
        
        for i, result in enumerate(results[:max_results], 1):
            # Format each result row
            result_str = ", ".join([f"{k}: {v}" for k, v in result.items()])
            text += f"{i}. {result_str}\n"
        
        if len(results) > max_results:
            text += f"\n... and {len(results) - max_results} more results"
        
        return text

    # ================= Generate answer from query results =================
    def generate_answer_from_query(self, question, cypher_query, results, estimated_hops):
        """Generate natural language answer from structured query results."""
        if not results:
            return "❌ No results found in the knowledge graph for this query."
        
        results_text = self.format_query_results(results, max_results=100)
        
        prompt = f"""
You are answering a knowledge graph question. The question was converted to a Cypher query and executed.

QUESTION: {question}
ESTIMATED HOPS NEEDED: {estimated_hops}

CYPHER QUERY EXECUTED:
{cypher_query}

QUERY RESULTS:
{results_text}

INSTRUCTIONS:
1. Provide a clear, concise answer based on the query results
2. For counting questions, count the results
3. List key results (top 10-20 if many)
4. Mention the hop count used in your reasoning
5. Mark: ✅ if results found, ❌ if no results

ANSWER:
"""
        
        answer = self._call_llm(prompt, max_tokens=500)
        return answer

    # ================= Node matching (fallback) =================
    def find_matching_nodes(self, entity_name, limit=5):
        """Return list of candidate nodes for an entity name."""
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
        return self.run_cypher(query_fuzzy, {"name": entity_name, "limit": limit})

    # ================= Fallback: Traditional subgraph approach =================
    def get_subgraph_fallback(self, start_node_ids, max_depth=3, max_nodes=200):
        """Fallback method using traditional subgraph traversal."""
        if not start_node_ids:
            return {"nodes": {}, "edges": [], "total_nodes": 0, "total_edges": 0}
        
        query = f"""
        MATCH (start)
        WHERE elementId(start) IN $node_ids
        WITH collect(start) as starts
        UNWIND starts as start
        OPTIONAL MATCH path = (start)-[*1..{max_depth}]-(connected)
        WITH starts + collect(DISTINCT connected) as all_nodes, 
             collect(DISTINCT path) as all_paths
        
        UNWIND all_nodes as n
        WITH collect(DISTINCT n) as unique_nodes, all_paths
        
        UNWIND all_paths as path
        WITH unique_nodes, relationships(path) as path_rels
        UNWIND path_rels as rel
        WITH unique_nodes, collect(DISTINCT rel) as unique_rels
        
        WITH unique_nodes[0..{max_nodes}] as limited_nodes, 
             unique_rels[0..{max_nodes * 3}] as limited_rels
        
        RETURN 
            [n IN limited_nodes | {{
                id: elementId(n),
                name: coalesce(n.name, n.title, toString(elementId(n))),
                labels: labels(n)
            }}] as nodes,
            [r IN limited_rels | {{
                type: type(r),
                start: coalesce(startNode(r).name, toString(elementId(startNode(r)))),
                end: coalesce(endNode(r).name, toString(elementId(endNode(r))))
            }}] as relationships
        """
        
        try:
            result = self.run_cypher(query, {"node_ids": start_node_ids})
            if not result or not result[0].get("nodes"):
                return {"nodes": {}, "edges": [], "total_nodes": 0, "total_edges": 0}
            
            rec = result[0]
            return {
                "nodes": rec.get("nodes", []),
                "edges": rec.get("relationships", []),
                "total_nodes": len(rec.get("nodes", [])),
                "total_edges": len(rec.get("relationships", []))
            }
        except:
            return {"nodes": {}, "edges": [], "total_nodes": 0, "total_edges": 0}

    # ================= Main pipeline =================
    def ask(self, question, max_depth=None, max_nodes_per_entity=200, auto_depth=True, use_cypher_generation=True):
        """
        Main question-answering pipeline.
        
        Args:
            question: The question to answer
            max_depth: Maximum number of hops (used only if cypher generation fails)
            max_nodes_per_entity: Maximum nodes to retrieve
            auto_depth: Automatically estimate required depth
            use_cypher_generation: If True, generate specific Cypher query (RECOMMENDED)
        
        Returns:
            Dictionary with question, answer, and stats
        """
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract entities
        entities = self.extract_entities(question)
        if not entities:
            return {
                "question": question,
                "entities": [],
                "answer": "❌ No specific entities found. Please rephrase with specific names.",
                "method": "none"
            }
        
        # Step 2: Estimate complexity
        estimated_hops = self.analyze_question_complexity(question)
        if max_depth is None:
            max_depth = estimated_hops
        
        # Step 3: Try Cypher generation approach (BETTER)
        if use_cypher_generation:
            print("\n[APPROACH] Using Cypher Query Generation")
            try:
                cypher_query = self.generate_cypher_query(question, entities)
                results = self.execute_generated_query(cypher_query)
                
                if results is not None:
                    answer = self.generate_answer_from_query(
                        question, cypher_query, results, estimated_hops
                    )
                    return {
                        "question": question,
                        "entities": entities,
                        "cypher_query": cypher_query,
                        "results": results,
                        "answer": answer,
                        "method": "cypher_generation",
                        "stats": {
                            "estimated_hops": estimated_hops,
                            "result_count": len(results) if results else 0
                        }
                    }
                else:
                    print("[!] Cypher generation failed, falling back to subgraph approach")
            except Exception as e:
                print(f"[!] Cypher generation error: {e}, falling back")
        
        # Step 4: Fallback to subgraph approach
        print("\n[APPROACH] Using Subgraph Traversal (Fallback)")
        all_node_ids = []
        for entity in entities:
            matches = self.find_matching_nodes(entity, limit=2)
            all_node_ids.extend([m['id'] for m in matches[:1]])
        
        if not all_node_ids:
            return {
                "question": question,
                "entities": entities,
                "answer": f"❌ Could not find nodes matching: {entities}",
                "method": "subgraph_failed"
            }
        
        subgraph = self.get_subgraph_fallback(all_node_ids, max_depth=max_depth)
        
        if subgraph['total_nodes'] == 0:
            return {
                "question": question,
                "entities": entities,
                "answer": "❌ No subgraph data retrieved",
                "method": "subgraph_failed"
            }
        
        # Format and answer
        context_text = f"Nodes: {subgraph['total_nodes']}, Edges: {subgraph['total_edges']}\n"
        context_text += f"Sample nodes: {[n['name'] for n in subgraph['nodes'][:20]]}\n"
        sample_edges = [f"{e['start']}-[{e['type']}]->{e['end']}" for e in subgraph['edges'][:30]]
        context_text += f"Sample edges: {sample_edges}"
        
        answer = f"⚠️ Using fallback subgraph method with {max_depth} hops. Results may be imprecise. Consider enabling cypher_generation=True."
        
        return {
            "question": question,
            "entities": entities,
            "subgraph": subgraph,
            "answer": answer,
            "method": "subgraph_fallback",
            "stats": {
                "estimated_hops": estimated_hops,
                "max_depth_used": max_depth,
                "total_nodes": subgraph['total_nodes']
            }
        }

    # ================= Visualization =================
    def visualize_bfs_result(self, result):
        """Print formatted result."""
        print("\n" + "="*60)
        print("GRAPH RAG RESULTS")
        print("="*60)
        
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Entities: {result.get('entities', [])}")
        
        if "cypher_query" in result:
            print(f"\nCypher Query:\n{result['cypher_query']}")
        
        if "stats" in result:
            stats = result["stats"]
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")

    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
        print("[+] Neo4j connection closed")