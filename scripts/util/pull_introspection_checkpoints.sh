#!/bin/bash
# Pull introspection checkpoints from Modal volume and assemble into JSONL.
# Run this after generation completes (or to check progress mid-run).
#
# Usage:
#   bash pull_introspection_checkpoints.sh           # check count only
#   bash pull_introspection_checkpoints.sh --pull     # download and assemble

set -e

echo "=== Introspection Checkpoint Status ==="

REFL_COUNT=$(modal volume ls foundry-adapters /introspection-checkpoints/reflections/ 2>/dev/null | wc -l | tr -d ' ')
DIAL_COUNT=$(modal volume ls foundry-adapters /introspection-checkpoints/dialogues/ 2>/dev/null | wc -l | tr -d ' ')

echo "Reflections: $REFL_COUNT / 3000"
echo "Dialogues:   $DIAL_COUNT / 100"

if [ "$1" = "--pull" ]; then
    echo ""
    echo "Pulling checkpoints from Modal volume..."

    mkdir -p data/training/introspection/checkpoints/reflections
    mkdir -p data/training/introspection/checkpoints/dialogues

    modal volume get foundry-adapters \
        introspection-checkpoints/reflections/ \
        data/training/introspection/checkpoints/reflections/

    modal volume get foundry-adapters \
        introspection-checkpoints/dialogues/ \
        data/training/introspection/checkpoints/dialogues/

    echo ""
    echo "Assembling JSONL files..."

    # Assemble reflections
    python3 -c "
import json, glob
files = sorted(glob.glob('data/training/introspection/checkpoints/reflections/*.json'))
with open('data/training/introspection/reflections.jsonl', 'w') as out:
    for f in files:
        with open(f) as inp:
            out.write(inp.read().strip() + '\n')
print(f'Assembled {len(files)} reflections → data/training/introspection/reflections.jsonl')
"

    # Assemble dialogues
    python3 -c "
import json, glob
files = sorted(glob.glob('data/training/introspection/checkpoints/dialogues/*.json'))
with open('data/training/introspection/dialogues.jsonl', 'w') as out:
    for f in files:
        with open(f) as inp:
            out.write(inp.read().strip() + '\n')
print(f'Assembled {len(files)} dialogues → data/training/introspection/dialogues.jsonl')
"

    echo ""
    echo "Done. Run scripts/data/filter_introspection.py next."
fi
