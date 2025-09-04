# RainGPM

A Python package for downloading, processing, and analyzing Global Precipitation Measurement (GPM) satellite data for hydrological applications.

## Overview

RainGPM provides tools to work with NASA's GPM IMERG (Integrated Multi-satellitE Retrievals for GPM) precipitation data. The package handles the entire workflow from data acquisition to processing and analysis:

- Download GPM satellite data from NASA servers
- Process raw data into usable formats
- Generate precipitation time series for specific locations
- Create visualizations of rainfall patterns

## Installation

### Requirements

- Python 3.8 or higher
- GDAL 3.4.3
- Other dependencies listed in requirements.txt

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/raingpm.git
cd raingpm
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv hidrogeo_env
source hidrogeo_env/bin/activate  # On Windows: hidrogeo_env\Scripts\activate
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

## Usage

### Basic Pipeline

The package provides a pipeline for downloading and processing GPM data:

```python
from datetime import datetime, timedelta, timezone
from pipeline import data_ingestion, data_prepare, data_processing

# Define date range (last 7 days)
dt_end = datetime.now(timezone.utc)
dt_init = dt_end - timedelta(days=7)
date_range = (dt_init, dt_end)

# Run the pipeline
data_ingestion(date_range)
data_prepare(date_range)
data_processing(date_range)
```

### Example: Visualizing Precipitation Data

```python
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# Open a processed raster file
file_path = "/path/to/processed/gpm/data.tif"
dataset = gdal.Open(file_path)
band = dataset.GetRasterBand(1)
raster_array = band.ReadAsArray()

# Mask no-data values
raster_array = np.ma.masked_equal(raster_array, band.GetNoDataValue())

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(raster_array.flatten(), bins=range(11), color='blue', alpha=0.7)
plt.title("Precipitation Distribution")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Frequency")
plt.show()
```

## Project Structure

```
raingpm/
├── __pycache__/
├── build/
├── config.py              # Configuration setup
├── examples/              # Example notebooks
├── hidrogeo_env/          # Virtual environment (not tracked)
├── misc/                  # Utility functions
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

## Data Sources

This package works with NASA's GPM IMERG data, which can be accessed at:
https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/

You'll need NASA Earth Data credentials to download the data.

## License

[Add your license information here]

## Author

Pedro Silva

## Acknowledgments

- NASA Global Precipitation Measurement Mission
- [Add any other acknowledgments here]
