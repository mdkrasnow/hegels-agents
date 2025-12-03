#!/bin/bash
# 
# Evaluation Data Accumulation Helper Script
# 
# This script provides convenient commands for accumulating evaluation data
# and running statistical validation on Hegel's agents.
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="$SCRIPT_DIR/accumulated_evaluation_data.json"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3."
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_DIR/accumulate_evaluation_data.py" ]; then
        print_error "accumulate_evaluation_data.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    if [ ! -d "$SCRIPT_DIR/src" ]; then
        print_error "src directory not found. Are you in the correct directory?"
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    add <n>           Add n evaluation rounds to existing data (5 questions per round)
    fresh <n>         Start fresh with n evaluation rounds  
    analyze           Run statistical analysis on existing data
    status            Show current data summary
    small-batch       Add 2 rounds (10 evaluations) - good for testing
    medium-batch      Add 5 rounds (25 evaluations) - good for regular collection
    large-batch       Add 10 rounds (50 evaluations) - good for substantial data
    full-analysis     Run large batch + comprehensive statistical analysis

Examples:
    $0 add 3              # Add 3 rounds (15 evaluations) to existing data
    $0 fresh 10           # Start fresh with 10 rounds (50 evaluations)  
    $0 analyze            # Analyze existing accumulated data
    $0 status             # Show summary of current data
    $0 small-batch        # Quick test: add 10 evaluations
    $0 medium-batch       # Regular collection: add 25 evaluations
    $0 full-analysis      # Comprehensive: 50 new evaluations + analysis

Data file: $DATA_FILE
EOF
}

run_command() {
    local cmd="$1"
    shift
    
    print_header "Running: $cmd"
    if $PYTHON_CMD "$SCRIPT_DIR/accumulate_evaluation_data.py" "$@"; then
        print_success "Command completed successfully"
    else
        print_error "Command failed"
        exit 1
    fi
}

case "${1:-}" in
    "add")
        if [ -z "$2" ]; then
            print_error "Please specify number of rounds to add"
            echo "Usage: $0 add <rounds>"
            exit 1
        fi
        print_header "Adding $2 evaluation rounds to existing data"
        run_command "add" --run-count "$2" --data-file "$DATA_FILE" --questions-per-round 5 -v
        ;;
        
    "fresh")
        if [ -z "$2" ]; then
            print_error "Please specify number of rounds for fresh start"
            echo "Usage: $0 fresh <rounds>"
            exit 1
        fi
        if [ -f "$DATA_FILE" ]; then
            print_warning "Backing up existing data file..."
            cp "$DATA_FILE" "${DATA_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        fi
        print_header "Starting fresh with $2 evaluation rounds"
        run_command "fresh" --run-count "$2" --output-file "$DATA_FILE" --questions-per-round 5 -v
        ;;
        
    "analyze")
        if [ ! -f "$DATA_FILE" ]; then
            print_error "No data file found at $DATA_FILE"
            print_warning "Run '$0 add <n>' or '$0 fresh <n>' first to collect data"
            exit 1
        fi
        print_header "Running statistical analysis"
        run_command "analyze" --analyze-only --data-file "$DATA_FILE" --analysis-output-dir "analysis_results"
        ;;
        
    "status")
        if [ ! -f "$DATA_FILE" ]; then
            print_warning "No data file found at $DATA_FILE"
            echo "Run '$0 add <n>' or '$0 fresh <n>' to start collecting data"
        else
            print_header "Current Data Summary"
            $PYTHON_CMD -c "
import json
from pathlib import Path

try:
    with open('$DATA_FILE', 'r') as f:
        data = json.load(f)
    
    total = data['metadata']['total_evaluations']
    unique = len(data['metadata']['unique_questions_evaluated'])
    sessions = len(data['metadata']['evaluation_sessions'])
    
    print(f'Total Evaluations: {total}')
    print(f'Unique Questions: {unique}')  
    print(f'Sessions: {sessions}')
    print(f'Ready for Analysis: {\"Yes\" if total >= 10 else f\"No (need {10-total} more)\"}')
    
    if total > 0:
        results = data['test_results']
        single_scores = [r['single_agent_quality_score'] for r in results]
        hegel_scores = [r['dialectical_quality_score'] for r in results] 
        improvements = [h - s for h, s in zip(hegel_scores, single_scores)]
        positive = sum(1 for imp in improvements if imp > 0)
        
        print()
        print(f'Avg Single Score: {sum(single_scores)/len(single_scores):.1f}')
        print(f'Avg Hegel Score: {sum(hegel_scores)/len(hegel_scores):.1f}')
        print(f'Avg Improvement: {sum(improvements)/len(improvements):.1f} pts')
        print(f'Positive Rate: {positive/total*100:.1f}% ({positive}/{total})')
        
except Exception as e:
    print(f'Error reading data: {e}')
    exit(1)
"
        fi
        ;;
        
    "small-batch")
        print_header "Small Batch: Adding 2 rounds (10 evaluations)"
        run_command "small-batch" --run-count 2 --data-file "$DATA_FILE" --questions-per-round 5 -v
        ;;
        
    "medium-batch") 
        print_header "Medium Batch: Adding 5 rounds (25 evaluations)"
        run_command "medium-batch" --run-count 5 --data-file "$DATA_FILE" --questions-per-round 5 -v
        ;;
        
    "large-batch")
        print_header "Large Batch: Adding 10 rounds (50 evaluations)"  
        run_command "large-batch" --run-count 10 --data-file "$DATA_FILE" --questions-per-round 5 -v
        ;;
        
    "full-analysis")
        print_header "Full Analysis: Large batch + comprehensive statistical analysis"
        run_command "full-analysis-collect" --run-count 10 --data-file "$DATA_FILE" --questions-per-round 5 --analyze --analysis-output-dir "comprehensive_analysis_results" -v
        ;;
        
    "help"|"--help"|"-h"|"")
        show_usage
        ;;
        
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

# Show final status for non-analyze commands
if [ "${1:-}" != "analyze" ] && [ "${1:-}" != "status" ]; then
    echo ""
    print_header "Final Status"
    $0 status
fi