#!/bin/sh
# Start-Script für Liquid Foundation Model Pipeline

echo "🌊 Liquid Foundation Model Pipeline"
echo "==================================="
echo ""

# Prüfe Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python3 nicht gefunden!"
    exit 1
fi

# Installation Check
echo "📦 Prüfe Dependencies..."
python3 -c "import flask" 2>/dev/null || {
    echo "Installing Flask..."
    pip3 install flask flask-cors
}

# Optionen
MODE=${1:-demo}

case $MODE in
    demo)
        echo "🎯 Starte Demo..."
        python3 liquid_pipeline.py
        ;;
    
    api)
        echo "🚀 Starte API Server..."
        python3 liquid_api_server.py --port 5000
        ;;
    
    test)
        echo "🧪 Teste Pipeline..."
        python3 -c "
from liquid_pipeline import LiquidPipeline
pipeline = LiquidPipeline()

# Test-Anfragen
queries = [
    'Was kostet ein Personalausweis?',
    'Wo kann ich mich ummelden?',
    'Wie lange dauert eine Baugenehmigung?'
]

print('\\n📋 Test-Anfragen:\\n')
for q in queries:
    result = pipeline.process(q)
    print(f'❓ {q}')
    print(f'💬 {result.adapted_response}')
    print(f'   ✓ Confidence: {result.confidence:.0%} | Time: {result.processing_time_ms:.1f}ms')
    print()
"
        ;;
    
    benchmark)
        echo "⚡ Performance Benchmark..."
        python3 -c "
import time
from liquid_pipeline import LiquidPipeline

pipeline = LiquidPipeline(enable_logging=False)
query = 'Was kostet ein Personalausweis?'

# Warmup
for _ in range(10):
    pipeline.process(query)

# Benchmark
print('\\n📊 Benchmark Results:\\n')

for n in [10, 100, 1000]:
    start = time.time()
    for _ in range(n):
        pipeline.process(query)
    elapsed = time.time() - start
    
    print(f'{n:4d} queries: {elapsed:6.2f}s = {n/elapsed:7.1f} queries/sec')

print('\\n✅ Benchmark complete!')
"
        ;;
    
    knowledge)
        echo "📚 Wissen hinzufügen..."
        echo "Entity name: "
        read ENTITY
        echo "Pattern type (cost/process/location/documents/duration/deadline): "
        read PATTERN
        echo "Value: "
        read VALUE
        
        python3 -c "
from liquid_pipeline import LiquidPipeline
pipeline = LiquidPipeline()
pipeline.add_knowledge('$ENTITY', '$PATTERN', '$VALUE')
print('✅ Wissen hinzugefügt!')
"
        ;;
    
    stats)
        echo "📈 Pipeline Statistiken..."
        python3 -c "
from liquid_pipeline import LiquidPipeline
pipeline = LiquidPipeline()
stats = pipeline.get_statistics()

print('\\n📊 Statistiken:\\n')
print(f'Gesamt Anfragen: {stats[\"total_queries\"]}')
print(f'Erfolgsrate: {stats[\"success_rate\"]:.1f}%')
print(f'Ø Verarbeitungszeit: {stats[\"avg_processing_time_ms\"]:.2f}ms')
print(f'Ø Confidence: {stats[\"avg_confidence\"]:.0%}')

if stats['top_patterns']:
    print('\\nTop Patterns:')
    for pattern, count in stats['top_patterns']:
        print(f'  - {pattern}: {count}')

if stats['top_entities']:
    print('\\nTop Entities:')
    for entity, count in stats['top_entities']:
        print(f'  - {entity}: {count}')
"
        ;;
    
    *)
        echo "Usage: $0 [demo|api|test|benchmark|knowledge|stats]"
        echo ""
        echo "Modes:"
        echo "  demo      - Run interactive demo"
        echo "  api       - Start API server"
        echo "  test      - Run basic tests"
        echo "  benchmark - Performance benchmark"
        echo "  knowledge - Add new knowledge"
        echo "  stats     - Show statistics"
        exit 1
        ;;
esac