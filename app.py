import os
import sys
from flask import Flask, render_template, request, jsonify
import time
from src.generate_mistral import generate_answer
from src.retrieve import RAGRetriever
from src.model_manager import ModelManager
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

app = Flask(__name__)

# Pre-load model on startup
print("🔄 Pre-loading model...")
model, tokenizer = ModelManager.get_model_and_tokenizer()
rag = RAGRetriever()

# Response cache
response_cache = {}
CACHE_TIMEOUT = 300  # 5 minutes

def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ''
    error = ''
    processing_time = 0
    
    if request.method == 'POST':
        start_time = time.time()
        query = request.form['query'].strip()
        
        # Check cache first
        cache_key = get_cache_key(query)
        if cache_key in response_cache:
            cache_time, cached_answer = response_cache[cache_key]
            if time.time() - cache_time < CACHE_TIMEOUT:
                answer = cached_answer
                processing_time = time.time() - start_time
                return render_template('index.html', answer=answer, error=error, processing_time=round(processing_time, 2))
        
        try:
            # Fast retrieval
            print(f"🔍 Retrieving results for: {query}")
            results = rag.retrieve(query)
            df = results["dataframe"]
            item_count = results["requested_items"]
            
            # Generate answer
            print("🤖 Generating answer...")
            answer = generate_answer(query, df, model, tokenizer, item_count)
            
            # Cache the result
            response_cache[cache_key] = (time.time(), answer)
            
            # Clean old cache entries
            clean_cache()
            
        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"❌ Error: {e}")
        
        processing_time = time.time() - start_time
    
    return render_template('index.html', answer=answer, error=error, processing_time=round(processing_time, 2))

def clean_cache():
    """Remove cache entries older than timeout"""
    current_time = time.time()
    keys_to_remove = []
    for key, (cache_time, _) in response_cache.items():
        if current_time - cache_time > CACHE_TIMEOUT:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del response_cache[key]

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# Async version for better performance
@app.route('/api/search', methods=['POST'])
def api_search():
    """Faster API endpoint for search"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    cache_key = get_cache_key(query)
    if cache_key in response_cache:
        cache_time, cached_answer = response_cache[cache_key]
        if time.time() - cache_time < CACHE_TIMEOUT:
            return jsonify({"answer": cached_answer, "cached": True})
    
    try:
        results = rag.retrieve(query)
        df = results["dataframe"]
        item_count = results["requested_items"]
        
        answer = generate_answer(query, df, model, tokenizer, item_count)
        
        # Cache result
        response_cache[cache_key] = (time.time(), answer)
        
        return jsonify({"answer": answer, "cached": False})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Pre-warm the system
    print("🔥 Pre-warming the system...")
    try:
        # Test retrieval with a simple query
        test_results = rag.retrieve("test")
        print("✅ System ready!")
    except Exception as e:
        print(f"⚠️ Pre-warm warning: {e}")
    
    app.run(threaded=True)