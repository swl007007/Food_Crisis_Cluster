# Claude Code Commands

**Title:** GeoRF Claude Code Command Documentation
**Purpose:** Document Claude Code specific automation and commands for GeoRF
**Scope:** Claude Code IDE integration and automation
**Owners:** GeoRF Development Team
**Last-Updated:** 2025-10-12

## Available Commands

### Development Commands
Commands for common development tasks with GeoRF framework.

#### Model Training Commands
```bash
# Train GeoRF model with specific configuration
/train-georf --start_year 2023 --end_year 2024 --forecasting_scope 1

# Train XGBoost model with batch processing
/train-xgboost --batch --start_year 2023 --end_year 2024

# Run baseline comparisons
/run-baselines --all
```

#### Testing Commands
```bash
# Run comprehensive test suite
/test-all

# Run performance benchmarks
/test-performance

# Run memory leak tests
/test-memory-leaks
```

#### Documentation Commands
```bash
# Generate API documentation
/docs-generate-api

# Update CLAUDE.md with latest changes
/docs-update-claude

# Validate documentation links
/docs-validate-links
```

### Utility Commands

#### Environment Setup
```bash
# Setup development environment
/setup-dev-env

# Install dependencies
/install-deps

# Validate configuration
/validate-config
```

#### Data Management
```bash
# Clean result directories
/cleanup-results --force

# Backup model checkpoints
/backup-checkpoints

# Verify data integrity
/verify-data
```

#### Memory Management
```bash
# Force memory cleanup
/cleanup-memory --pipeline gf

# Monitor memory usage
/monitor-memory

# Kill hanging processes
/kill-hanging-processes
```

## Command Implementation

### Command Structure
Claude Code commands should follow this pattern:
```python
def command_train_georf(args):
    """Train GeoRF model with specified parameters.

    Args:
        args: Parsed command line arguments

    Usage:
        /train-georf --start_year 2023 --end_year 2024 --forecasting_scope 1
    """
    # Implementation here
    pass
```

### Error Handling
All commands should include proper error handling:
```python
try:
    result = execute_training(args)
    print(f"SUCCESS: Training completed - {result}")
except Exception as e:
    print(f"ERROR: Training failed - {e}")
    return 1
```

### Progress Reporting
Commands should provide clear progress feedback:
```python
def report_progress(current, total, operation):
    """Report progress for long-running operations."""
    percentage = (current / total) * 100
    print(f"Progress: {percentage:.1f}% - {operation}")
```

## Integration with GeoRF

### Batch Processing Integration
Commands should integrate with existing batch processing:
```python
def command_run_batch_training(pipeline_type, scope_range):
    """Run batch training with memory management."""

    if pipeline_type == "georf":
        subprocess.run(["run_georf_batches.bat"])
    elif pipeline_type == "xgboost":
        subprocess.run(["run_xgboost_batches.bat"])

    # Monitor and report progress
    monitor_batch_progress()
```

### Configuration Integration
Commands should respect configuration settings:
```python
import config

def validate_command_config():
    """Validate configuration before running commands."""
    required_params = ['MIN_DEPTH', 'MAX_DEPTH', 'N_JOBS']

    for param in required_params:
        if not hasattr(config, param):
            raise ValueError(f"Missing required config parameter: {param}")
```

### Result Management
Commands should properly manage results:
```python
def save_command_results(command_name, results, timestamp):
    """Save command results with proper naming."""

    result_file = f"command_results_{command_name}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            'command': command_name,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)
```

## Command Categories

### Training Commands
- `/train-georf`: Train Random Forest GeoRF model
- `/train-xgboost`: Train XGBoost GeoRF model
- `/train-batch`: Run batch training pipeline
- `/train-2layer`: Train 2-layer model architecture

### Evaluation Commands
- `/eval-model`: Evaluate trained model performance
- `/compare-models`: Compare multiple model results
- `/baseline-comparison`: Run 4-model comparison analysis
- `/generate-plots`: Create performance visualization plots

### Development Commands
- `/setup-issue`: Create new issue structure from templates
- `/validate-code`: Run code quality checks
- `/update-docs`: Update documentation and examples
- `/run-tests`: Execute test suites

### Maintenance Commands
- `/cleanup-all`: Comprehensive cleanup of temporary files
- `/backup-models`: Backup trained models and checkpoints
- `/monitor-resources`: Monitor system resource usage
- `/validate-environment`: Check development environment setup

## Usage Examples

### Basic Model Training
```bash
# Quick training run for testing
/train-georf --start_year 2024 --end_year 2024 --forecasting_scope 1

# Production training with batch processing
/train-batch --pipeline georf --full-production
```

### Development Workflow
```bash
# Setup new feature development
/setup-issue --type feature --name "adjacency-matrix-optimization"

# Validate changes before commit
/validate-code --check-style --run-tests

# Update documentation after changes
/update-docs --include-api --validate-links
```

### Troubleshooting
```bash
# Clean up after failed run
/cleanup-all --force --include-cache

# Monitor system resources
/monitor-resources --duration 300 --interval 10

# Validate data integrity
/verify-data --check-missing --validate-format
```

## Command Development Guidelines

### Adding New Commands
1. **Follow Naming Convention**: Use descriptive verb-noun pattern
2. **Include Help Text**: Provide clear usage documentation
3. **Error Handling**: Implement robust error handling and recovery
4. **Progress Reporting**: Show progress for long-running operations
5. **Result Validation**: Validate command results and outputs

### Testing Commands
- All commands should be tested with valid and invalid inputs
- Include integration tests with GeoRF framework
- Test error conditions and edge cases
- Validate command output and side effects

### Documentation Requirements
- Each command needs usage documentation
- Include examples of common usage patterns
- Document expected inputs and outputs
- Provide troubleshooting guidance

**Source:** Claude Code best practices and automation patterns
**Source:** GeoRF workflow analysis and common operations