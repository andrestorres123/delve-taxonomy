# Delve: AI-Powered Taxonomy Generation

Delve is a production-ready SDK and CLI for automatically generating taxonomies from your data using state-of-the-art language models.

📚 **[Read the full documentation →](https://wildcampstudio.mintlify.app)**

## Quick Start

### Installation

```bash
pip install delve-taxonomy

# Set API keys
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # Required for classifier embeddings
```

### CLI

```bash
# Basic usage (shows progress spinners)
delve run data.csv --text-column text

# With progress bars and ETA
delve run data.csv --text-column text -v

# Quiet mode (errors only)
delve run data.csv --text-column text -q

# JSON with nested data
delve run data.json --json-path "$.messages[*].content"
```

### Python SDK

```python
from delve import Delve, Verbosity

# Initialize client (silent by default - library best practice)
delve = Delve()

# Or with progress output
delve = Delve(verbosity=Verbosity.NORMAL)

# Run taxonomy generation
result = delve.run_sync("data.csv", text_column="text")

# Access results
print(f"Generated {len(result.taxonomy)} categories")
for category in result.taxonomy:
    print(f"  - {category.name}: {category.description}")

# Access labeled documents
for doc in result.labeled_documents[:5]:
    print(f"  [{doc.category}] {doc.content[:50]}...")
```

### Binary Detection (Single Category)

For fast filtering when you know the category you're looking for:

```python
from delve import Delve

# Find all refund-related documents (~$1-2 for 30K docs, runs in minutes)
result = Delve.find_matches(
    "data.csv",
    category={
        "name": "Refund Request",
        "description": "User asking for refund or money back",
        "keywords": ["refund", "money back", "cancel"],
    },
    text_column="text",
    threshold=0.6,
)

print(f"Found {result.stats['matches']} matches")
for doc in result.matched_documents[:5]:
    print(f"  [{doc.confidence:.2f}] {doc.content[:50]}...")
```

## Features

- **Automated Taxonomy Generation** - No manual category creation using Claude 3.5 Sonnet
- **Binary Detection** - Fast, cheap single-category filtering with `find_matches()`
- **Multiple Data Sources** - CSV, JSON/JSONL, LangSmith runs, pandas DataFrames
- **Smart Categorization** - Iterative refinement with minibatch clustering
- **Flexible Exports** - JSON, CSV, and Markdown reports

## Requirements

- Python 3.9+
- Anthropic API key (for taxonomy generation)
- OpenAI API key (for classifier embeddings when sample_size > 0)

## Documentation

- 📖 [Full Documentation](https://wildcampstudio.mintlify.app)
- 🚀 [Quickstart Guide](https://wildcampstudio.mintlify.app/quickstart)
- 💻 [CLI Reference](https://wildcampstudio.mintlify.app/cli-reference)
- 🐍 [SDK Reference](https://wildcampstudio.mintlify.app/sdk-reference)
- 📚 [Examples](https://wildcampstudio.mintlify.app/examples)

## Development

```bash
# Install dependencies
uv sync

# Run tests
pytest tests/

# Run linting
ruff check src/

# Format code
ruff format src/
```

### Documentation Development

To work on the documentation locally, you'll need Node.js 20.17+ (for Mintlify):

```bash
# If using nvm, the project includes .nvmrc
nvm use

# Install Mintlify CLI (if not already installed)
npm install -g mintlify

# Run the docs server
cd docs
mintlify dev
```

See the [full documentation](https://wildcampstudio.mintlify.app) for more details on contributing and development.
