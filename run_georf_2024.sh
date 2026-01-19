#!/bin/bash
# Run GeoRF for 2024 with all 12 months

DESIRED_TERMS="2024-10"

echo "Running GeoRF for 2024 (all 12 months)..."
echo "DESIRED_TERMS: $DESIRED_TERMS"
echo ""

/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe app/main_model_GF.py --start_year 2024 --end_year 2024 --forecasting_scope 1 --desired_terms "$DESIRED_TERMS"
