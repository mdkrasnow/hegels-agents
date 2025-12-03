# Evaluation Data Accumulation Guide

This guide explains how to use the evaluation data accumulation system to build up a robust dataset for statistical validation of Hegel's agents.

## Quick Start

The easiest way to get started is using the shell script:

```bash
# Check current status  
./run_evaluation_accumulation.sh status

# Add 10 evaluations (2 rounds of 5)
./run_evaluation_accumulation.sh small-batch

# Add 25 evaluations (5 rounds of 5)  
./run_evaluation_accumulation.sh medium-batch

# Run statistical analysis on accumulated data
./run_evaluation_accumulation.sh analyze
```

## Commands Overview

### Data Collection Commands

| Command | Description | Evaluations Added |
|---------|-------------|------------------|
| `small-batch` | Quick test batch | 10 (2 rounds × 5) |
| `medium-batch` | Regular collection | 25 (5 rounds × 5) |
| `large-batch` | Substantial collection | 50 (10 rounds × 5) |
| `add <n>` | Add n rounds | n × 5 |
| `fresh <n>` | Start fresh with n rounds | n × 5 |

### Analysis Commands

| Command | Description |
|---------|-------------|
| `analyze` | Run statistical analysis on existing data |
| `status` | Show current data summary |
| `full-analysis` | Large batch + comprehensive analysis |

## Recommended Workflow

### 1. Start Small (Testing)
```bash
# Start with a small batch to test everything works
./run_evaluation_accumulation.sh small-batch

# Check the results
./run_evaluation_accumulation.sh status
```

### 2. Regular Data Collection
```bash
# Run medium batches regularly to build up data
./run_evaluation_accumulation.sh medium-batch

# Check progress
./run_evaluation_accumulation.sh status
```

### 3. Statistical Analysis
```bash
# Once you have 30+ evaluations, run analysis
./run_evaluation_accumulation.sh analyze

# Check results in: analysis_results/
```

## Understanding the Output

### Data Summary
- **Total Evaluations**: Number of single vs dialectical comparisons
- **Unique Questions**: How many different questions have been evaluated  
- **Sessions**: Number of evaluation rounds run
- **Ready for Analysis**: Whether you have enough data (≥10 recommended)

### Quick Stats
- **Avg Single Score**: Average quality score for single-agent responses
- **Avg Hegel Score**: Average quality score for dialectical responses  
- **Avg Improvement**: Mean improvement (Hegel - Single)
- **Positive Rate**: Percentage of evaluations where Hegel performed better

## File Locations

- **Data File**: `accumulated_evaluation_data.json` - Contains all evaluation results
- **Analysis Results**: `analysis_results/` - Statistical analysis output
- **Backup Files**: `*.backup.*` - Automatic backups when starting fresh

## Advanced Usage

### Using the Python Script Directly

```bash
# Add specific number of evaluation rounds
python3 accumulate_evaluation_data.py --run-count 3 --data-file accumulated_evaluation_data.json

# Start fresh with custom settings
python3 accumulate_evaluation_data.py --run-count 5 --output-file my_results.json --questions-per-round 3

# Analyze existing data with custom parameters  
python3 accumulate_evaluation_data.py --analyze-only --data-file my_results.json --confidence-level 0.99

# Run evaluation and analysis in one command
python3 accumulate_evaluation_data.py --run-count 8 --analyze --analysis-output-dir detailed_analysis/
```

### Question Selection Strategies

- `--question-selection random` (default): Randomly sample questions each round
- `--question-selection sequential`: Cycle through questions systematically  
- `--question-selection all`: Use all questions in order

### Configuration Options

- `--confidence-level`: Statistical confidence level (default: 0.95)
- `--significance-level`: P-value threshold (default: 0.05) 
- `--random-seed`: For reproducible results (default: 42)
- `--corpus-dir`: Custom corpus directory location

## Sample Sizes for Statistical Power

| Sample Size | Statistical Power | Recommendation |
|-------------|------------------|----------------|
| 10-19 | Low | Minimum for basic analysis |
| 20-29 | Moderate | Good for preliminary conclusions |
| 30-49 | Good | Recommended for reliable results |
| 50+ | Excellent | High confidence conclusions |

## Troubleshooting

### "Evaluation infrastructure not available"
- Check that you're in the correct directory
- Verify `src/` directory exists
- Ensure configuration files are set up properly

### "No data file found"  
- Run `./run_evaluation_accumulation.sh add 1` to start collecting data
- Check that you're in the project root directory

### Analysis shows "insufficient evidence"
- Collect more data with additional evaluation rounds
- Aim for 30+ evaluations for reliable statistical analysis
- Check if there's actually improvement in your quick stats

## Tips

1. **Start Small**: Begin with `small-batch` to verify everything works
2. **Regular Collection**: Run `medium-batch` weekly to build up data gradually
3. **Monitor Progress**: Use `status` command to track your progress
4. **Save Backups**: The system automatically backs up when you start fresh
5. **Patience**: Each evaluation takes time - run batches when you can let it run unattended

## Example Complete Workflow

```bash
# Day 1: Initial test
./run_evaluation_accumulation.sh small-batch
./run_evaluation_accumulation.sh status

# Day 2: Add more data  
./run_evaluation_accumulation.sh medium-batch
./run_evaluation_accumulation.sh status

# Day 3: First analysis
./run_evaluation_accumulation.sh analyze

# Day 4: Add more data for better power
./run_evaluation_accumulation.sh medium-batch

# Day 5: Comprehensive analysis
./run_evaluation_accumulation.sh analyze
```

This approach builds up a robust dataset over time while providing regular feedback on progress and results.