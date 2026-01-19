# Documentation Migration Notice

**Date:** 2025-01-17
**Status:** Documentation has been reorganized into `.ai/` directory

## Migration Summary

All documentation from this `docs/` folder has been migrated to the structured `.ai/` directory for better organization and AI assistant integration.

## Where Did Files Go?

### Architecture Documentation
- `GeoRF_Framework_Documentation.md` → `.ai/docs/architecture/framework-overview.md`
- `Architecture.md` → `.ai/docs/architecture/system-architecture.md`

### Feature Documentation
- `Feature_Test_Doc/feature_shortlist.txt` → `.ai/docs/features/feature-catalog.md` (converted to markdown)
- `Feature_Test_Doc/varlist.xlsx` → `.ai/docs/features/variable-definitions.xlsx`

### Guides and Workflows
- `Feature_Test_Doc/QUICK_START_VISUAL_DEBUG.md` → `.ai/issues/issue-0001-memory-allocation-visual-debug/docs/problem-statement.md`
- `Feature_Test_Doc/README_MONTHLY_PROCESSING.md` → `.ai/issues/issue-0001-memory-allocation-visual-debug/docs/solution-monthly-processing.md`
- `Feature_Test_Doc/README_VISUAL_DEBUG_BATCHES.md` → `.ai/docs/guides/visual-debug-yearly-batches.md`
- `Feature_Test_Doc/ARCHIVE_SYSTEM_README.md` → `.ai/docs/guides/archive-system.md`

### Technical Assessment Reports
- `Technical_Assesment_Doc/Class_Imbalance_Analysis_Report.md` → `.ai/issues/issue-0002-class-imbalance-f1-discrepancy/docs/analysis-report.md`
- `Technical_Assesment_Doc/Partition_stall_diagnostics.md` → `.ai/issues/issue-0003-partition-stalling/docs/diagnostics-report.md`
- `Technical_Assesment_Doc/Partition_Enclaves_Assessment.md` → `.ai/issues/issue-0004-enclave-polygons/docs/assessment-report.md`
- `Technical_Assesment_Doc/partition_accuracy_diagnosis.md` → `.ai/issues/issue-0005-bipartitioned-appearance/docs/diagnosis-report.md`
- `Technical_Assesment_Doc/xgb_vs_gf_parity_report.md` → `.ai/issues/issue-0007-georf-geoxgb-parity/docs/parity-verification-report.md`

### Test Documentation
- `Test_Doc/FINAL_IMPLEMENTATION_STATUS.md` → `.ai/issues/issue-0006-monthly-evaluation-refactoring/docs/implementation-status.md`
- `Test_Doc/REFACTORING_STATUS.md` → `.ai/issues/issue-0006-monthly-evaluation-refactoring/docs/refactoring-status.md`
- `Test_Doc/VISUAL_DEBUG_REMAINING_CHANGES.md` → `.ai/issues/issue-0006-monthly-evaluation-refactoring/docs/remaining-changes.md`

### Test Outputs
- `test_run_log_0917/*` → `.ai/issues/issue-0001-memory-allocation-visual-debug/testing/test-run-outputs/`

## New Structure Benefits

1. **Issue-Based Organization**: All related documents for a problem/feature are in one place
2. **Clear Status Tracking**: Each issue has explicit status (RESOLVED, DIAGNOSED, IN_PROGRESS, etc.)
3. **Better Discoverability**: Consistent folder structure makes finding information easier
4. **Comprehensive Documentation**: Nothing lost, everything reorganized logically
5. **AI-Friendly**: Structured for AI assistant context and understanding

## How to Find Documentation Now

### Primary Entry Point
**Start here:** `.ai/ai-context-structure.md`
- Complete overview of new structure
- File mapping reference
- Issue summary table

### Quick Navigation

**For Architecture Information:**
```
.ai/docs/architecture/
├── framework-overview.md      # Comprehensive GeoRF guide
├── system-architecture.md     # High-level design
└── architecture-insights.md   # Deep technical insights
```

**For Feature Information:**
```
.ai/docs/features/
├── feature-catalog.md          # All features with descriptions
└── variable-definitions.xlsx   # Variable metadata
```

**For Usage Guides:**
```
.ai/docs/guides/
├── visual-debug-quick-start.md
├── visual-debug-yearly-batches.md
├── visual-debug-monthly-processing.md  # RECOMMENDED for memory issues
└── archive-system.md
```

**For Issue Tracking:**
```
.ai/issues/
├── issue-0001-memory-allocation-visual-debug/     # RESOLVED
├── issue-0002-class-imbalance-f1-discrepancy/     # EXPLAINED
├── issue-0003-partition-stalling/                 # DIAGNOSED
├── issue-0004-enclave-polygons/                   # EXPLAINED
├── issue-0005-bipartitioned-appearance/           # EXPLAINED
├── issue-0006-monthly-evaluation-refactoring/     # IN_PROGRESS (7/11 files)
├── issue-0007-georf-geoxgb-parity/                # COMPLETED
└── issue-0008-validation-coverage-gaps/           # PROPOSED
```

## Files Remaining in docs/

Some files remain in `docs/` for reference:
- **Metadata files**: `_MIGRATION_NOTICE.md`, `_migration_index.csv`
- **Supplementary data**: JSON reports, verification artifacts
- **Legacy references**: Files not yet migrated or kept for backward compatibility

## Questions?

For the complete file mapping and detailed migration information, see:
- `.ai/ai-context-structure.md` - Complete structure guide
- `CLAUDE.md` - Primary project documentation (updated with new references)

---

**Migration Completed:** 2025-01-17
**Issues Created:** 8 (0001-0008)
**Files Migrated:** 40+ documents
**Structure:** Issue-based organization with comprehensive READMEs
