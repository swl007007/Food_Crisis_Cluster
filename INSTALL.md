# Installation Guide

## System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Disk Space**: 2GB+ for dependencies and model outputs

## Installation Methods

### Method 1: Using pip (Recommended for most users)

1. **Create a virtual environment** (recommended):
   ```bash
   python3.12 -m venv georf-env

   # On Windows:
   georf-env\Scripts\activate

   # On Linux/Mac:
   source georf-env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Using conda

1. **Create conda environment from file**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate georf-food-crisis
   ```

### Method 3: Manual installation

If you prefer to install packages individually:

```bash
# Core packages
pip install numpy pandas polars scipy

# Machine learning
pip install scikit-learn xgboost imbalanced-learn

# Visualization
pip install matplotlib seaborn

# Geospatial
pip install geopandas shapely contextily

# Other utilities
pip install Pillow shap tqdm
```

## Windows-Specific Setup

If using Windows with the specific Python path mentioned in the documentation:

```bash
C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe -m pip install -r requirements.txt
```

## Verification

Verify your installation by running:

```bash
python -c "import numpy, pandas, sklearn, xgboost, geopandas; print('All core packages imported successfully!')"
```

## Troubleshooting

### Common Issues

**1. GeoPandas installation fails:**
```bash
# Try installing GDAL first (conda is easier for this)
conda install -c conda-forge geopandas
```

**2. XGBoost installation issues:**
```bash
# Make sure you have updated pip
pip install --upgrade pip
pip install xgboost
```

**3. Shapely installation problems:**
```bash
# Use conda-forge channel
conda install -c conda-forge shapely
```

**4. Memory issues during model training:**
- Use the batch processing scripts (`run_georf_batches.bat`, `run_xgboost_batches.bat`)
- These scripts handle memory cleanup between iterations

### Platform-Specific Notes

**Windows:**
- Some geospatial packages may require Microsoft Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Linux:**
- May need to install GDAL development libraries:
  ```bash
  sudo apt-get install gdal-bin libgdal-dev
  ```

**macOS:**
- Install GDAL via Homebrew:
  ```bash
  brew install gdal
  ```

## Testing Your Installation

Run a quick test with the demo script:

```bash
python demo/GeoRF_demo.py
```

Or run the main model with test parameters:

```bash
python app/main_model_GF.py --start_year 2024 --end_year 2024 --forecasting_scope 1
```

## Updating Dependencies

To update all packages to their latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

## Development Setup

If you're developing or contributing to the codebase:

```bash
# Install development dependencies
pip install -r requirements.txt

# Install in editable mode (if setup.py exists)
pip install -e .
```

## Additional Resources

- **Documentation**: See `CLAUDE.md` for detailed project documentation
- **Architecture**: See `docs/Architecture.md` for system design
- **Technical Details**: See `docs/GeoRF_Framework_Documentation.md`

## Support

If you encounter issues not covered here:
1. Check the documentation in the `docs/` directory
2. Review error messages carefully - they often indicate missing dependencies
3. Ensure you're using Python 3.12 or higher
