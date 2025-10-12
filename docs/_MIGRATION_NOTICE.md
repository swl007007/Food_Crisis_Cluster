# Documentation Migration Notice

**Migration Date:** 2025-10-12
**Status:** COMPLETED

## Documentation Reorganization

The GeoRF documentation has been reorganized to provide better structure for both users and developers. This directory (`./docs/`) now serves as a reference archive, while active documentation has been moved to more appropriate locations.

## New Documentation Structure

### For Users - Getting Started
- **CLAUDE.md** (root directory) - Primary usage guide
  - Running models and batch processing
  - Configuration and troubleshooting
  - Architecture overview and examples

### For Developers - Contributing
- **.ai/** directory - Structured development documentation
  - **.ai/docs/** - Foundation standards and constraints
  - **.ai/templates/** - Templates for new work
  - **.ai/{issue-id}/** - Organized development work packages

### For AI Assistants - Development Support
- **.claude/commands/** - Claude Code specific automation
- **.ai/ai-context-structure.md** - Complete structure guide

## Migration Summary

| Original File | New Location | Purpose |
|---------------|--------------|---------|
| `GeoRF_Framework_Documentation.md` | Multiple locations (see index) | Split into user guide (CLAUDE.md) and developer docs (.ai/) |
| `Architecture.md` | `.ai/{issue-id}/design/architecture.md` | Moved to example design document |
| `Technical_Design.md` | `.ai/{issue-id}/design/architecture.md` | Integrated into architecture design |
| `Technical_Assesment_Doc/` | `.ai/templates/` patterns | Used as templates for analysis documents |

## Finding Information

### If You Need...
- **To run GeoRF models** → See `CLAUDE.md` in root directory
- **Development standards** → See `.ai/docs/` directory
- **To start new development** → See `.ai/templates/README.md`
- **Migration details** → See `_migration_index.csv` in this directory

### Quick Navigation
```bash
# Primary user documentation
cat ../CLAUDE.md

# Development workflow guide
cat ../.ai/ai-context-structure.md

# Complete migration index
cat _migration_index.csv

# Documentation relationships
cat _link_graph.md
```

## Legacy Files

The original files in this directory are preserved for reference:
- Content remains unchanged for historical continuity
- Links and references are maintained for backup
- Migration stubs are added to guide users to new locations

## Support

If you cannot find the information you need:
1. Check the migration index (`_migration_index.csv`)
2. Review the link graph (`_link_graph.md`)
3. Search the new structure starting with `CLAUDE.md`

**Last Updated:** 2025-10-12