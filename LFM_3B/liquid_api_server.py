#!/usr/bin/env python3
"""
Liquid Foundation Model API Server
RESTful API für die Liquid Pipeline
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Import unsere Pipeline
from liquid_pipeline import LiquidPipeline, QueryResult, PatternType

app = Flask(__name__)
CORS(app)  # Enable CORS für Frontend-Integration

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Pipeline Instance
pipeline = None

def init_pipeline():
    """Initialisiert die Pipeline"""
    global pipeline
    pipeline = LiquidPipeline(knowledge_db_path="municipal_knowledge.db")
    logger.info("Liquid Pipeline initialized")

@app.route('/health', methods=['GET'])
def health_check():
    """Health Check Endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Liquid Foundation Model API",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/query', methods=['POST'])
def process_query():
    """Hauptendpoint für Anfragen"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request"
            }), 400
        
        query = data['query']
        context = data.get('context', {})
        
        # Verarbeite Anfrage
        result = pipeline.process(query, context)
        
        # Konvertiere Result zu JSON
        response = {
            "query_id": result.query_id,
            "original_query": result.original_query,
            "pattern_type": result.pattern_type,
            "entity": result.entity,
            "confidence": result.confidence,
            "response": result.adapted_response,
            "success": result.success,
            "processing_time_ms": result.processing_time_ms,
            "timestamp": result.timestamp
        }
        
        if not result.success:
            response["error"] = result.error_message
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def process_batch():
    """Batch-Verarbeitung mehrerer Anfragen"""
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                "error": "Missing 'queries' field in request"
            }), 400
        
        queries = data['queries']
        context = data.get('context', {})
        
        # Verarbeite alle Anfragen
        results = []
        for query in queries:
            result = pipeline.process(query, context)
            results.append({
                "query": query,
                "response": result.adapted_response,
                "confidence": result.confidence,
                "success": result.success
            })
        
        return jsonify({
            "results": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/knowledge', methods=['POST'])
def add_knowledge():
    """Fügt neues Wissen zur Wissensbasis hinzu"""
    try:
        data = request.get_json()
        
        required_fields = ['entity', 'pattern_type', 'value']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}"
                }), 400
        
        entity = data['entity']
        pattern_type = data['pattern_type']
        value = data['value']
        metadata = data.get('metadata', {})
        
        # Füge Wissen hinzu
        pipeline.add_knowledge(entity, pattern_type, value, metadata)
        
        return jsonify({
            "success": True,
            "message": f"Knowledge added: {entity} - {pattern_type}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding knowledge: {str(e)}")
        return jsonify({
            "error": "Failed to add knowledge",
            "message": str(e)
        }), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Gibt Statistiken über die Pipeline zurück"""
    try:
        stats = pipeline.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500

@app.route('/patterns', methods=['GET'])
def get_patterns():
    """Gibt verfügbare Pattern-Typen zurück"""
    patterns = [p.value for p in PatternType]
    return jsonify({
        "patterns": patterns,
        "total": len(patterns)
    })

@app.route('/entities', methods=['GET'])
def get_entities():
    """Gibt bekannte Entities zurück"""
    entities = list(pipeline.entity_extractor.entities.keys())
    return jsonify({
        "entities": entities,
        "total": len(entities)
    })

# WebSocket Support für Real-Time Updates (optional)
try:
    from flask_socketio import SocketIO, emit
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('query')
    def handle_realtime_query(data):
        """Real-time Query Processing via WebSocket"""
        query = data.get('query', '')
        context = data.get('context', {})
        
        result = pipeline.process(query, context)
        
        emit('response', {
            'query': query,
            'response': result.adapted_response,
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time_ms
        })
    
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    logger.warning("flask-socketio not installed, WebSocket support disabled")

# Beispiel HTML Interface
@app.route('/')
def index():
    """Simple Web Interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liquid Municipal API</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            input, button { padding: 10px; margin: 5px; }
            #query { width: 400px; }
            #response { 
                margin-top: 20px; 
                padding: 20px; 
                background: #f0f0f0; 
                border-radius: 5px;
                min-height: 100px;
            }
            .context-controls { margin: 20px 0; }
            .slider { width: 200px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌊 Liquid Municipal Model</h1>
            
            <div>
                <input type="text" id="query" placeholder="Was kostet ein Personalausweis?" />
                <button onclick="sendQuery()">Fragen</button>
            </div>
            
            <div class="context-controls">
                <h3>Kontext-Einstellungen:</h3>
                <label>Formalität: <input type="range" class="slider" id="formality" min="0" max="1" step="0.1" value="0.5" /></label><br/>
                <label>Dringlichkeit: <input type="range" class="slider" id="urgency" min="0" max="1" step="0.1" value="0.3" /></label><br/>
                <label>Sprachlevel: <input type="range" class="slider" id="language_level" min="0" max="1" step="0.1" value="1.0" /></label>
            </div>
            
            <div id="response"></div>
            
            <script>
                async function sendQuery() {
                    const query = document.getElementById('query').value;
                    const context = {
                        formality: parseFloat(document.getElementById('formality').value),
                        urgency: parseFloat(document.getElementById('urgency').value),
                        language_level: parseFloat(document.getElementById('language_level').value)
                    };
                    
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query, context})
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('response').innerHTML = `
                        <strong>Antwort:</strong> ${data.response}<br/>
                        <small>Confidence: ${(data.confidence * 100).toFixed(0)}% | 
                        Zeit: ${data.processing_time_ms.toFixed(1)}ms</small>
                    `;
                }
                
                // Enter-Taste Support
                document.getElementById('query').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendQuery();
                });
            </script>
        </div>
    </body>
    </html>
    '''

def create_cli():
    """CLI für API Server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Liquid Foundation Model API Server')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--db-path', type=str, default='municipal_knowledge.db', 
                       help='Path to knowledge database')
    
    return parser

if __name__ == '__main__':
    parser = create_cli()
    args = parser.parse_args()
    
    # Initialize pipeline with custom DB path if provided
    pipeline = LiquidPipeline(knowledge_db_path=args.db_path)
    
    print(f"""
    🌊 Liquid Foundation Model API Server
    =====================================
    
    Starting server on http://{args.host}:{args.port}
    
    Endpoints:
    - GET  /              - Web Interface
    - GET  /health        - Health Check
    - POST /query         - Single Query
    - POST /batch         - Batch Queries
    - POST /knowledge     - Add Knowledge
    - GET  /statistics    - Get Statistics
    - GET  /patterns      - List Pattern Types
    - GET  /entities      - List Known Entities
    
    Press Ctrl+C to stop
    """)
    
    if SOCKETIO_AVAILABLE:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)