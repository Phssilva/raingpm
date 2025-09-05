# RainGPM

A Python package for downloading, processing, and analyzing Global Precipitation Measurement (GPM) satellite data for hydrological applications.

## Overview

RainGPM provides tools to work with NASA's GPM IMERG (Integrated Multi-satellitE Retrievals for GPM) precipitation data. The package handles the entire workflow from data acquisition to processing and analysis:

- Download GPM satellite data from NASA servers
- Process raw data into usable formats
- Generate precipitation time series for specific locations
- Create visualizations of rainfall patterns
- Perform spatial analysis with custom regions of interest

## Installation

### Requirements

- Python 3.8 or higher
- GDAL 3.4.3
- Key dependencies:
  - rasterio 1.4.2
  - geopandas 1.0.1
  - numpy 2.0+
  - pandas 2.2.3
  - xarray 2024.10.0
  - dask 2024.10.0
  - scipy 1.14.1
  - matplotlib 3.9.2
  - scikit-learn 1.4.2
  - pyspark 3.5.4
  - dynaconf 3.2.6
  - h5py 3.12.1

For a complete list of dependencies, see `requirements.txt`.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Phssilva/raingpm.git
cd raingpm
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv raingpm_env
source raingpm_env/bin/activate  # On Windows: raingpm_env\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

The package uses [Dynaconf](https://www.dynaconf.com/) for configuration management. Settings are stored in:

- `settings.toml` - Main configuration file
- `.secrets.toml` - Credentials and sensitive information (not tracked by git)

You'll need to create a `.secrets.toml` file with your NASA Earth Data credentials:

```toml
[sec.GPM]
password = "your_nasa_earthdata_password"
```

### Storage Configuration

The package uses several directories for data storage, which are defined in `settings.toml`:

```toml
[STORAGE]
gpm_raw = "/path/to/raw/data"
gpm_intermediary = "/path/to/intermediary/data"
gpm_prepared = "/path/to/prepared/data"
gpm_processed = "/path/to/processed/data"
extent_gpm_template = "/path/to/shapefile/template.shp"
raster_gpm_template = "/path/to/raster/template.tif"
```

## Usage

### Complete Pipeline

The package provides a pipeline for downloading, preparing, and processing GPM data:

```python
from datetime import datetime, timedelta
from raingpm.pipeline import data_ingestion, data_prepare, data_processing

# Define date range (last 7 days)
dt_end = datetime.now()
dt_end = dt_end - timedelta(hours=3)  # Adjust for timezone if needed
dt_init = dt_end - timedelta(days=7)
date_range = (dt_init, dt_end)

# Run the pipeline
r = data_ingestion(date_range)
if r:  # Only proceed if data was successfully downloaded
    data_prepare(date_range)
    data_processing(date_range)
```

### Step-by-Step Usage

#### 1. Download GPM Data

```python
from datetime import datetime, timedelta
from raingpm.pipeline import data_ingestion

# Define date range
dt_end = datetime(2024, 1, 18)
dt_init = dt_end - timedelta(days=7)
date_range = (dt_init, dt_end)

# Download data
data_ingestion(date_range)
```

#### 2. Prepare GPM Data

```python
from raingpm.pipeline import data_prepare

# Prepare downloaded data
data_prepare(date_range)
```

#### 3. Process GPM Data

```python
from raingpm.pipeline import data_processing

# Process prepared data
data_processing(date_range)
```

### Working with Rasters

The package includes utilities for working with raster data, including creating rasters from shapefiles:

```python
from raingpm.misc.geo.raster import create_raster_from_shapefile

# Create a raster from a shapefile
shapefile_path = "path/to/shapefile.shp"
output_raster = "path/to/output.tif"
create_raster_from_shapefile(shapefile_path, output_raster)
```

**Note:** When creating rasters from shapefiles with geographic CRS (like EPSG:4326), the pixel size is automatically converted from meters to degrees. For latitude, 1 degree ≈ 111,320 meters at the equator, and for longitude, it varies with latitude (111,320 * cos(latitude) meters).

### Example: Visualizing Precipitation Data

```python
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Open a processed raster file
file_path = "/path/to/processed/gpm/data.tif"
with rasterio.open(file_path) as src:
    raster_array = src.read(1)
    nodata = src.nodata
    
    # Mask no-data values
    raster_array = np.ma.masked_equal(raster_array, nodata)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = show(raster_array, ax=ax, cmap='Blues', title='Precipitation (mm)')
    plt.colorbar(im, label='Precipitation (mm)')
    plt.savefig('precipitation_map.png', dpi=300)
    plt.show()
```

## Project Structure

```
raingpm/
├── __pycache__/
├── build/
├── config.py              # Configuration setup
├── examples/              # Example scripts and notebooks
│   ├── example.ipynb
│   └── generate_raster_from_shp.py
├── misc/                  # Utility functions
│   ├── geo/               # Geospatial utilities
│   ├── __init__.py
│   └── files.py           # File handling utilities
├── pipeline.py            # Main processing pipeline
├── requirements.txt       # Dependencies
├── settings.toml          # Configuration settings
├── setup.py               # Package installation
├── src/                   # Source code
│   ├── DownloadGPMData.py # Download GPM data
│   ├── PrepareGPMData.py  # Prepare raw data
│   └── ProcessingGPMData.py # Process prepared data
└── storage/               # Data storage directory
```

## Data Processing Workflow

1. **Download (DownloadGPMData.py)**: 
   - Downloads GPM IMERG data from NASA servers
   - Supports both EARLY and LATE data products
   - Uses parallel processing for efficient downloads

2. **Prepare (PrepareGPMData.py)**:
   - Extracts precipitation data from HDF5 files
   - Crops data to region of interest using a shapefile template
   - Converts to GeoTIFF format with proper projection

3. **Process (ProcessingGPMData.py)**:
   - Aggregates hourly data into daily, monthly, or custom periods
   - Performs statistical analysis on precipitation patterns
   - Generates final output products for analysis

## Data Sources

Register at https://gpm.nasa.gov/data/directory to get your NASA Earth Data credentials.

This package works with NASA's GPM IMERG data, which can be accessed at:
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/

You'll need NASA Earth Data credentials to download the data.

## License

[Add your license information here]

## Author

Pedro Silva

## Acknowledgments

- NASA Global Precipitation Measurement Mission
