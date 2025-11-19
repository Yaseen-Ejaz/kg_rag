from flask import Flask, request, jsonify
from flask_cors import CORS
from graph_rag_system import GraphRAGSystem
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Graph RAG System
rag_system = None

def get_rag_system():
    """Get or create RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = GraphRAGSystem(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            openai_api_key=config.OPENAI_API_KEY,
            openai_model=config.OPENAI_MODEL
        )
    return rag_system


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "BFS-Based Graph RAG System API",
        "version": "1.0.0"
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Main endpoint to ask questions to the Graph RAG system
    
    Request body:
    {
        "question": "Your question here",
        "max_depth": 2,  // required from frontend
        "max_nodes_per_entity": 100  // required from frontend
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400
        
        question = data['question']
        
        # These parameters must now come from frontend
        if 'max_depth' not in data or 'max_nodes_per_entity' not in data:
            return jsonify({
                "error": "Missing 'max_depth' or 'max_nodes_per_entity' in request body"
            }), 400
        
        max_depth = int(data['max_depth'])
        max_nodes_per_entity = int(data['max_nodes_per_entity'])
        
        # Validate ranges
        if max_depth < 1 or max_depth > 10:
            return jsonify({
                "error": "max_depth must be between 1 and 10"
            }), 400
        
        if max_nodes_per_entity < 10 or max_nodes_per_entity > 1000:
            return jsonify({
                "error": "max_nodes_per_entity must be between 10 and 1000"
            }), 400
        
        # Get RAG system
        rag = get_rag_system()
        
        # Process question
        result = rag.ask(
            question=question,
            max_depth=max_depth,
            max_nodes_per_entity=max_nodes_per_entity
        )
        
        return jsonify({
            "success": True,
            "data": result
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Get the graph database schema"""
    try:
        rag = get_rag_system()
        schema = rag.get_schema()
        
        return jsonify({
            "success": True,
            "data": schema
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/extract-entities', methods=['POST'])
def extract_entities():
    """
    Extract entities from a question
    
    Request body:
    {
        "question": "Your question here"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400
        
        question = data['question']
        rag = get_rag_system()
        entities = rag.extract_entities(question)
        
        return jsonify({
            "success": True,
            "data": {
                "question": question,
                "entities": entities
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/find-nodes', methods=['POST'])
def find_nodes():
    """
    Find matching nodes for an entity
    
    Request body:
    {
        "entity": "Entity name"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'entity' not in data:
            return jsonify({
                "error": "Missing 'entity' field in request body"
            }), 400
        
        entity = data['entity']
        rag = get_rag_system()
        nodes = rag.find_matching_nodes(entity)
        
        return jsonify({
            "success": True,
            "data": {
                "entity": entity,
                "matching_nodes": nodes
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/bfs-subgraph', methods=['POST'])
def bfs_subgraph():
    """
    Get BFS subgraph from a starting node
    
    Request body:
    {
        "start_node": "Node name",
        "max_depth": 2,  // optional
        "max_nodes": 50  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'start_node' not in data:
            return jsonify({
                "error": "Missing 'start_node' field in request body"
            }), 400
        
        start_node = data['start_node']
        max_depth = data.get('max_depth', 7)
        max_nodes = data.get('max_nodes', 300)
        
        rag = get_rag_system()
        subgraph = rag.apoc_subgraph(start_node, max_depth, max_nodes)
        
        return jsonify({
            "success": True,
            "data": subgraph
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Close RAG system on app shutdown"""
    global rag_system
    if rag_system is not None:
        rag_system.close()
        rag_system = None

@app.route('/api/full-graph', methods=['GET'])
def get_full_graph():
    """
    Get the complete graph structure with all nodes and relationships
    
    Optional query parameters:
    - limit: maximum number of nodes (default 200)
    """
    try:
        limit = request.args.get('limit', 200, type=int)
        rag = get_rag_system()
        
        # Cypher query to get all nodes and relationships
        query = """
        MATCH (n)
        WITH n LIMIT $limit
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN 
            collect(DISTINCT {
                id: id(n), 
                name: coalesce(n.name, toString(id(n))), 
                labels: labels(n)
            }) as nodes,
            collect(DISTINCT {
                source: coalesce(startNode(r).name, toString(id(startNode(r)))),
                target: coalesce(endNode(r).name, toString(id(endNode(r)))),
                relationship: type(r)
            }) as relationships
        """
        
        result = rag.run_cypher(query, {"limit": limit})
        
        if result and len(result) > 0:
            nodes = result[0].get('nodes', [])
            relationships = result[0].get('relationships', [])
            
            # Filter out null relationships
            relationships = [r for r in relationships if r['source'] and r['target']]
            
            return jsonify({
                "success": True,
                "data": {
                    "nodes": nodes,
                    "edges": relationships,
                    "total_nodes": len(nodes),
                    "total_edges": len(relationships)
                }
            })
        else:
            return jsonify({
                "success": True,
                "data": {
                    "nodes": [],
                    "edges": [],
                    "total_nodes": 0,
                    "total_edges": 0
                }
            })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



if __name__ == '__main__':
    print("\n" + "="*60)
    print("BFS-BASED GRAPH RAG SYSTEM - FLASK API")
    print("="*60)
    print(f"Starting server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    print("="*60 + "\n")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )