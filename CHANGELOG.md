# Changelog

All notable changes to Delve will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-15

### Changed

- **BREAKING: `requires-python` is now `>=3.11`** (was `>=3.9`). The LangChain
  1.x stack requires Python 3.10+; 3.11 matches the tested/CI range.
- **Default models updated** to `anthropic/claude-opus-4-8` (main) and
  `anthropic/claude-haiku-4-5` (fast).
- **All LLM calls use structured outputs** (schema-enforced JSON) instead of
  regex/XML parsing, removing a class of silent parse failures.
- **TNT-LLM prompts are vendored in the repo** instead of being pulled from
  LangSmith Hub at runtime — no network access or LangSmith account needed.
- **Dependencies hardened**: the LangChain stack is pinned to the 1.x line and
  capped below the next major; `pandas` and `scikit-learn` capped below their
  next majors; unused `langchain-fireworks` and `langchain-community` removed.

### Added

- **Provider-aware API key validation** — required keys are derived from the
  `provider/model` strings, so an OpenAI-only run needs only `OPENAI_API_KEY`.
- **Reproducible builds**: committed `uv.lock`; CI installs with `uv sync --frozen`.
- **Release automation**: tag-triggered PyPI publishing via OIDC trusted
  publishing, guarded so the tag must match the package version.

### Fixed

- Full static-type pass: `mypy` is clean and now a blocking CI gate.

---

## [0.1.8] - 2025-12-16

### Changed

- **Updated default models to Claude 4.5**:
  - Main model: `claude-sonnet-4-5-20250929` (was `claude-3-5-sonnet-20241022`)
  - Fast model: `claude-haiku-4-5-20251001` (was `claude-3-haiku-20240307`)

### Documentation

- Updated SDK reference with comprehensive metadata structure documentation
- Added verbosity output examples to CLI reference
- Enhanced examples with metadata access and predefined taxonomy usage
- Added OpenAI API key requirement to setup instructions

---

## [0.1.7] - 2025-01-16

### Added

- **Enhanced metadata in results**: `result.metadata` now includes comprehensive run information:
  - **Timing**: `run_duration_seconds` - total processing time
  - **Category distribution**: `category_counts` - document count per category
  - **Classifier metrics**: `classifier_metrics` with train/test accuracy and F1 scores
  - **Labeling breakdown**: `llm_labeled_count`, `classifier_labeled_count`
  - **Source info**: `source` with type, path, and column names
  - **Quality tracking**: `skipped_document_count`, `warnings` list

### Example

```python
result = delve.run_sync("data.csv", text_column="text")
print(result.metadata)
# {
#     "num_documents": 5000,
#     "num_categories": 10,
#     "run_duration_seconds": 145.32,
#     "category_counts": {"Bug Fix": 1250, "Feature": 890, ...},
#     "classifier_metrics": {"test_f1": 0.847, "test_accuracy": 0.85, ...},
#     "llm_labeled_count": 100,
#     "classifier_labeled_count": 4900,
#     "source": {"type": "csv", "path": "data.csv", "text_column": "text"},
#     "warnings": [],
#     ...
# }
```

---

## [0.1.6] - 2025-01-16

### Added

- **DEBUG mode enhancements**: Added meaningful debug output to differentiate DEBUG from VERBOSE mode:
  - Full configuration display at startup (model, sample size, batch size, etc.)
  - Taxonomy category listing with IDs
  - Classifier class distribution and sample counts per category
  - Detailed train/test metrics

---

## [0.1.5] - 2025-01-16

### Fixed

- **Classifier class weight bug**: Fixed incorrect class weight mapping when training classifier with sparse category indices. Previously, when some taxonomy categories had no labeled examples (e.g., documents labeled as "Other" were skipped), the class weights were incorrectly mapped using sequential indices instead of actual class indices, causing `ValueError: The classes, [X], are not in class_weight`.

---

## [0.1.4] - 2025-01-16

### Added

- **New Console System**: Unified output management with `Console` class
- **Verbosity Levels**: Five levels of output control
  - `SILENT` (SDK default) - No output, ideal for library consumers
  - `QUIET` (`-q`) - Errors only
  - `NORMAL` (CLI default) - Spinners and completion checkmarks
  - `VERBOSE` (`-v`) - Progress bars with throughput-based ETA
  - `DEBUG` (`-vv`) - Full debug output including warnings
- **Progress Bars**: Real-time progress tracking with honest ETA based on observed throughput
- **Rich Integration**: Beautiful terminal output using the `rich` library

### Changed

- **CLI Flags**: Replaced `--verbose/--quiet` with `-q/-v/-vv` pattern (Unix standard)
- **SDK Default**: Now silent by default (library best practice)

### Fixed

- Removed debug print statements from exception handlers
- Warnings now only show in DEBUG mode (not cluttering normal output)
- **Early API key validation**: Both Anthropic and OpenAI keys are validated immediately at startup, before any processing begins. Users no longer wait 5+ minutes only to fail on a missing key

### Usage

```bash
# CLI
delve run data.csv --text-column text        # Normal (spinners)
delve run data.csv --text-column text -q     # Quiet (errors only)
delve run data.csv --text-column text -v     # Verbose (progress bars)
delve run data.csv --text-column text -vv    # Debug (everything)
```

```python
# SDK
from delve import Delve
from delve.console import Verbosity

delve = Delve()  # Silent by default
delve = Delve(verbosity=Verbosity.NORMAL)    # With output
delve = Delve(verbosity=Verbosity.VERBOSE)   # With progress bars
```

---

## [0.1.0] - 2024-01-15

### Initial Release

Delve v0.1.0 is the first production-ready release, transforming the taxonomy_generator project into a fully-featured SDK and CLI for AI-powered taxonomy generation.

### Added

#### Core Features
- **SDK API**: `Delve` class providing programmatic access to taxonomy generation
- **CLI Interface**: `delve` command-line tool built with Click
- **Multiple Data Sources**:
  - CSV files with configurable text and ID columns
  - JSON/JSONL files with JSONPath support for nested data
  - LangSmith projects with time-based filtering
  - Pandas DataFrames for in-memory processing
- **Adapter Pattern**: Pluggable data source architecture for easy extensibility
- **Multiple Output Formats**:
  - `taxonomy.json` - Machine-readable taxonomy with metadata
  - `labeled_documents.json` - Documents with assigned categories
  - `labeled_data.csv` - Spreadsheet-friendly format
  - `taxonomy_reference.csv` - Category lookup table
  - `report.md` - Human-readable summary with statistics
  - `metadata.json` - Run configuration and metadata

#### Processing Pipeline
- **Automated Workflow**: Non-interactive pipeline for production use
- **Smart Sampling**: Automatic sampling of large datasets for efficiency
- **Iterative Clustering**: Minibatch-based taxonomy generation
- **Quality Review**: Built-in LLM validation of taxonomy quality
- **Document Labeling**: Automatic categorization with explanations
- **Progress Tracking**: Real-time feedback during processing

#### Configuration
- Configurable LLM models (main and fast)
- Adjustable sample sizes and batch sizes
- Custom use case descriptions
- Flexible output directory and formats
- Verbose/quiet modes

#### Developer Experience
- Async and sync APIs
- Type hints throughout codebase
- Comprehensive documentation
- Working examples for all use cases
- Clear error messages and validation

### Changed

**Breaking Changes from Previous Version**:

- **Package Rename**: `react-agent` → `delve`
- **Mode**: Interactive (LangGraph Studio) → Non-interactive (SDK/CLI)
- **Interface**: Studio-based → Programmatic and CLI
- **State Structure**: Removed interactive fields (`messages`, `user_feedback`, `UserFeedback` class)
- **Configuration**: Removed `max_runs`, added `output_formats`, `output_dir`, `verbosity`, `use_case`
- **Graph Flow**: Removed human-in-the-loop nodes (`request_taxonomy_approval`, `handle_user_feedback`)

#### Architecture Changes
- Renamed core module from `src/react_agent/` to `src/delve/`
- Refactored nodes into `core/` directory
- Introduced `adapters/` for data source abstraction
- Introduced `exporters/` for output format generation
- Simplified graph: removed interrupts and conditional feedback routing

### Fixed

- **Missing Dependency**: Added `langsmith>=0.1.0` to dependencies (was used but not declared)
- **Import Paths**: Updated all imports from `react_agent` to `delve`

### Technical Details

#### Dependencies Added
- `click>=8.1.0` - CLI framework
- `pandas>=2.0.0` - Data manipulation
- `jsonpath-ng>=1.6.0` - JSON path expressions
- `rich>=13.0.0` - Enhanced CLI output
- `langsmith>=0.1.0` - LangSmith integration (previously missing)

#### Graph Changes
```
Old Flow (Interactive):
get_runs → summarize → get_minibatches → generate_taxonomy
→ update_taxonomy (loop) → review_taxonomy → request_taxonomy_approval
→ handle_user_feedback (INTERRUPT) → label_documents → END

New Flow (Automated):
load_data → summarize → get_minibatches → generate_taxonomy
→ update_taxonomy (loop) → review_taxonomy → label_documents
→ save_results → END
```

#### Project Structure
```
src/delve/
├── __init__.py              # Public API exports
├── client.py               # SDK client
├── result.py               # Result objects
├── configuration.py        # Configuration
├── state.py                # State definitions
├── graph.py                # LangGraph workflow
├── prompts.py              # LLM prompts
├── utils.py                # Utilities
├── routing.py              # Conditional routing
├── core/                   # Processing nodes
│   ├── data_loader.py
│   ├── summarizer.py
│   ├── batch_generator.py
│   ├── taxonomy_generator.py
│   ├── taxonomy_updater.py
│   ├── taxonomy_reviewer.py
│   ├── document_labeler.py
│   └── results_saver.py
├── adapters/               # Data source adapters
│   ├── base.py
│   ├── csv_adapter.py
│   ├── json_adapter.py
│   ├── langsmith_adapter.py
│   └── dataframe_adapter.py
├── exporters/              # Output generators
│   ├── base.py
│   ├── json_exporter.py
│   ├── csv_exporter.py
│   ├── markdown_exporter.py
│   └── metadata_exporter.py
└── cli/                    # CLI implementation
    ├── main.py
    └── utils.py
```

### Documentation

- **README.md**: Comprehensive documentation with installation, usage, and examples
- **examples/**: Working example scripts for all use cases
  - `basic_csv_example.py` - Simple CSV processing
  - `json_example.py` - JSON with JSONPath
  - `dataframe_example.py` - Pandas integration
  - `advanced_usage.py` - Advanced features
  - `sample_data.csv` - Test data
  - `README.md` - Examples documentation

### Known Limitations

- No hierarchical taxonomies (planned for v0.3.0)
- No confidence scores for categories (planned for v0.2.0)
- No custom prompt templates (planned for v0.3.0)
- No web UI (planned for v0.5.0)

### Migration Guide

If upgrading from the previous `react-agent` version:

1. **Update imports**:
   ```python
   # Old
   from react_agent import ...

   # New
   from delve import Delve
   ```

2. **Use new SDK/CLI interface**:
   ```python
   # Old (interactive)
   # Used with LangGraph Studio

   # New (programmatic)
   delve = Delve()
   result = delve.run_sync("data.csv", text_column="text")
   ```

3. **Note**: This release breaks backward compatibility. The interactive Studio-based workflow is no longer supported. For human-in-the-loop workflows, wait for v0.2.0 which will add optional iterative mode.

### Contributors

- [Your name]

### Acknowledgments

Built with LangChain and LangGraph. Powered by Anthropic's Claude models.

---

## Future Releases

### [0.2.0] - Planned
- Optional iterative mode with human feedback
- Confidence scores for categories
- Category merging suggestions

### [0.3.0] - Planned
- Hierarchical taxonomies
- Custom prompt templates
- Fine-tuned model support

### [0.4.0] - Planned
- S3 and database adapters
- REST API server
- Batch processing improvements

### [0.5.0] - Planned
- Web UI
- Visualization tools
- Annotation interfaces
