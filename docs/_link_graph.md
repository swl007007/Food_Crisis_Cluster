# Documentation Link Graph

**Title:** GeoRF Documentation Relationship Map
**Purpose:** Document relationships and cross-references between new and old documentation
**Last-Updated:** 2025-10-12

## Primary Documentation Hierarchy

```
CLAUDE.md (ROOT)
├── Usage & Running Instructions
├── Architecture Overview
├── Configuration Guide
└── Development Notes

.ai/ (STRUCTURED DEVELOPMENT)
├── ai-context-structure.md
├── docs/ (FOUNDATION)
│   ├── code-style-guide.md
│   ├── log-format-guide.md
│   ├── implementation-constraints.md
│   ├── requirements-management-constraints.md
│   ├── issue-structure-guide.md
│   └── concurrency-locks.md
├── templates/ (SCAFFOLDING)
│   └── README.md
└── {issue-id}/ (WORK PACKAGES)
    ├── {requirement}.md
    ├── design/architecture.md
    ├── implementation/task-breakdown.md
    ├── testing/unit-test-cases.md
    └── docs/prd.md

.claude/ (AUTOMATION)
└── commands/README.md

./docs/ (LEGACY REFERENCE)
├── [Original files preserved]
└── [Migration stubs added]
```

## Cross-Reference Map

### Primary Entry Points
- **CLAUDE.md** → Primary user documentation
- **.ai/ai-context-structure.md** → Development workflow guide
- **.ai/docs/** → Foundation standards for all development

### Internal References

#### CLAUDE.md References
```markdown
CLAUDE.md
├── References: config.py (configuration parameters)
├── References: src/ modules (architecture components)
├── References: batch scripts (execution)
└── Links to: Result directory structure documentation
```

#### .ai/docs/ Cross-References
```markdown
.ai/docs/code-style-guide.md
├── Referenced by: All development issues
├── Sources: ./docs/GeoRF_Framework_Documentation.md#development-notes
└── Links to: .ai/docs/implementation-constraints.md

.ai/docs/log-format-guide.md
├── Referenced by: Troubleshooting and debugging
├── Sources: ./docs/test_run_log_0917/ (example formats)
└── Links to: .ai/docs/code-style-guide.md

.ai/docs/implementation-constraints.md
├── Referenced by: All technical implementations
├── Sources: ./docs/GeoRF_Framework_Documentation.md#troubleshooting
└── Links to: .ai/docs/requirements-management-constraints.md

.ai/docs/requirements-management-constraints.md
├── Referenced by: All requirement specifications
├── Sources: Project management best practices
└── Links to: .ai/docs/issue-structure-guide.md

.ai/docs/issue-structure-guide.md
├── Referenced by: All issue creation
├── Sources: Software development methodology
└── Links to: .ai/templates/README.md

.ai/docs/concurrency-locks.md
├── Referenced by: Multi-user development scenarios
├── Sources: ./docs/GeoRF_Framework_Documentation.md#memory-management
└── Links to: .ai/docs/implementation-constraints.md
```

#### Template System References
```markdown
.ai/templates/README.md
├── Referenced by: Issue creation workflow
├── Links to: .ai/docs/issue-structure-guide.md
├── Links to: .ai/docs/requirements-management-constraints.md
└── Provides templates for: All issue types

.ai/{issue-id}/
├── Uses templates from: .ai/templates/
├── References: .ai/docs/ (for standards)
├── Links to: CLAUDE.md (for context)
└── Sources: Relevant ./docs/ sections
```

#### Example Issue References
```markdown
.ai/spatial-partitioning-optimization/
├── spatial-partitioning-performance.md
│   ├── Sources: ./docs/Architecture.md#validation-coverage-by-group
│   ├── Sources: ./docs/Technical_Design.md#partition-algorithm-details
│   └── References: .ai/docs/requirements-management-constraints.md
└── design/architecture.md
    ├── Sources: ./docs/Architecture.md (multiple sections)
    ├── Sources: ./docs/Technical_Design.md
    └── References: .ai/docs/implementation-constraints.md
```

## Legacy Documentation References

### Original Files → New Locations
```markdown
./docs/GeoRF_Framework_Documentation.md
├── Section 1-2: → CLAUDE.md (preserved)
├── Section 3: → .ai/spatial-partitioning-optimization/design/architecture.md (split)
├── Section 4: → .ai/docs/code-style-guide.md (patterns extracted)
├── Section 5-10: → CLAUDE.md (preserved)
└── Section 11: → .ai/docs/implementation-constraints.md (split)

./docs/Architecture.md
├── All sections: → .ai/spatial-partitioning-optimization/design/architecture.md
└── Reference stub: → ./docs/Architecture.md (migration pointer)

./docs/Technical_Design.md
├── Algorithm details: → .ai/spatial-partitioning-optimization/design/architecture.md
└── Reference stub: → ./docs/Technical_Design.md (migration pointer)
```

### Assessment Documents → Templates
```markdown
./docs/Technical_Assesment_Doc/
├── Class_Imbalance_Analysis_Report.md → .ai/templates/ (analysis template)
├── refactoring_report.json → .ai/docs/code-style-guide.md (patterns)
├── partition_accuracy_diagnosis.md → .ai/templates/ (debugging template)
├── Partition_Enclaves_Assessment.md → .ai/templates/ (assessment template)
├── partition_rounds_vs_final.md → .ai/templates/ (comparison template)
└── Partition_stall_diagnostics.md → .ai/templates/ (troubleshooting template)
```

## Navigation Patterns

### For Users (Getting Started)
1. Start with **CLAUDE.md** for usage instructions
2. Reference **config.py** for configuration
3. Use batch scripts for execution
4. Check **CLAUDE.md#troubleshooting** for issues

### For Developers (Contributing)
1. Read **.ai/ai-context-structure.md** for overview
2. Review **.ai/docs/** for standards and constraints
3. Create issue using **.ai/templates/**
4. Follow **.ai/docs/issue-structure-guide.md**

### For AI Assistants (Development Support)
1. Understand structure via **.ai/ai-context-structure.md**
2. Apply standards from **.ai/docs/**
3. Use templates from **.ai/templates/**
4. Reference legacy docs in **./docs/** for context

## Maintenance Guidelines

### Link Validation
- All internal links should resolve to valid files
- Cross-references should be bidirectional where appropriate
- Broken links should be reported in broken links report

### Update Propagation
- Changes to foundation documents (.ai/docs/) affect all issues
- Template updates may require migration of existing issues
- CLAUDE.md updates should be reflected in context documentation

### Consistency Checks
- Terminology should be consistent across all documents
- File naming should follow established conventions
- Source attribution should be maintained for all migrated content

## Dead Links (To Be Created)
- Templates in .ai/templates/ (beyond README.md)
- Additional example issues demonstrating different patterns
- Specific troubleshooting guides for common scenarios
- Performance benchmark documentation

**Source:** Analysis of documentation migration and reorganization
**Migration Date:** 2025-10-12