import os
from contextlib import contextmanager
from datetime import datetime

import rasterio.features
from rasterio.transform import from_origin
import dask.array as da
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import shapely
import h5py
from osgeo import gdal
from pandas import DataFrame
from rasterio import MemoryFile
from rasterio.profiles import DefaultGTiffProfile
from rasterstats import point_query
from rasterstats.io import Raster
from skimage.transform import resize
from scipy.ndimage import zoom
from shapely.geometry import Polygon, mapping, box
from shapely.geometry.point import Point
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, Affine
from rasterio.mask import mask
import rioxarray as rxr
import xarray as xr
from rasterio.merge import merge
from multiprocessing import Pool
from functools import partial
from loguru import logger
from ..files import Files


@contextmanager
def clip(raster, pol: Polygon):
    """Clip a raster with a polygon

    Parameters
    ----------
    raster : rasterio open dataset
        An input rasterio dataset to be clipped
    pol : Polygon
        Polygon with the georeferenced coordinates

    Yields
    -------
    rasterio DatasetReader
        the clipped raster
    """
    # use context manager so DatasetReader and MemoryFile get cleaned up automatically
    data, transform = rasterio.mask.mask(
        raster, [pol], crop=True, filled=True, all_touched=True
    )

    profile = raster.profile
    profile.update(
        transform=transform, driver="GTiff", height=data.shape[1], width=data.shape[2]
    )
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


#@logger_wraps(level="DEBUG")
@contextmanager
def sum_to_rasterio(profile: dict, paths: list, band: int = 1):
    """Accumulated a list of rasters to a rasterio datasetReader
    (sum all rasters of the band).

    Parameters
    ----------
    profile : dict
        the profile of the raster (metadata)
    paths : list
        A list of paths for rasters
    band : int, optional
        The number of the band, by default 1

    Yields
    -------
    rasterio DatasetReader
        the accumulated raster
    """
    profile.update(driver="GTiff", count=1)
    data = accumulate_bands(paths, band=band)
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


def np_to_raster_file(outfilename, overwrite_meta, np_raster, tmp_raster):
    with rasterio.open(tmp_raster) as src0:
        meta = src0.meta

    meta.update(overwrite_meta)

    with rasterio.open(outfilename, "w", **meta) as dst:
        dst.write_band(1, np_raster.astype(rasterio.float32))


def crop_roi(roi, input_file, outputname):
    """Crop Region of Interest.
    Based on GeoTiff ROI file, specific for each Transmission Line,
    it cut the input file (GeoTiff or Shapefile) and save it at the
    output destination.
    If the specific input file is empty, this function fulfill an
    empty information.

    Parameters
    ----------
    roi : GeoTiff
        Auxiliary data, containing TL`s ROI
    input_file : str
        Input filename with extension (.shp or .tif)
    outputname : str
        Output filename
    """
    base_roi = gdal.Open(roi)
    ulx, xres, xskew, uly, yskew, yres = base_roi.GetGeoTransform()
    lrx = ulx + (base_roi.RasterXSize * xres)
    lry = uly + (base_roi.RasterYSize * yres)
    polygon_ = Polygon([(ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry), (ulx, uly)])

    if os.path.splitext(os.path.basename(input_file))[1] == ".tif":
        input_file = gdal.Open(input_file)
        gdal.Translate(outputname, input_file, projWin=[ulx, uly, lrx, lry])
    elif os.path.splitext(os.path.basename(input_file))[1] == ".shp":
        shp = gpd.read_file(input_file)
        clipped = gpd.clip(shp, polygon_)
        if len(clipped.index) == 0:
            clipped.loc[0] = [0, 0, 0, 0, 0, None]
            clipped["geometry"] = clipped["geometry"].apply(lambda x: Point(1, 2))

        clipped.to_file(filename=outputname, driver="ESRI Shapefile")


def sum_rasters(df: DataFrame, category: str, out_dir: str):
    """Accumulated all tifs from df.path and save the final tif
    metadata into the database. This final tif will be used after
    for the classification of the landslide risky.

    Parameters
    ----------
    df : DataFrame
        Dataframe with the specifics files to be accumulated
    category : str
        One of these: 'l24h', 'l96h' and 'l360h'
    out_dir : str
        The output dir to save the accumulated rasters.
    id_tl : str
        The transmission Line uuid to insert into the table at the
        database.
    reference_datetime : datetime
        The reference datetime to search the files
    """

    logger.info("Accumulating rasters...")

    path_list = df["path"].values

    ref_dt = df["file_datetime"].max()
    start = datetime.strftime(df["file_datetime"].min(), "S%Y%m%d-%H%M%S")
    end = datetime.strftime(ref_dt, "E%Y%m%d-%H%M%S")

    outfilename = os.path.join(out_dir, f"{category}_{start}_{end}.tiff")
    tmp_name = os.path.join(out_dir, "tmp_.tiff")

    logger.info(
        f"Category={category} | Sum {len(path_list)} rasters between {start} and {end}"
    )

    if os.path.exists(outfilename):
        logger.info("Moving old file to temporary name...")
        os.rename(outfilename, tmp_name)

    meta = {"dtype:": "float32", "driver:": "GTiff", "count": 1}
    sum_to_file(path_list, outfilename, meta)

    if os.path.exists(outfilename) and os.path.exists(tmp_name):
        logger.info("Removing old file datalake with the same path...")
        os.remove(tmp_name)

def create_raster_from_shapefile(shapefile_path, output_path, pixel_size=5000):
    """
    Cria um raster com pixels de tamanho específico a partir de um shapefile.
    
    Parameters:
    -----------
    shapefile_path : str
        Caminho para o arquivo shapefile
    output_path : str
        Caminho para salvar o raster de saída
    pixel_size : float
        Tamanho do pixel em metros (padrão: 5000 para pixels de 5km)
    """
    # Carrega o shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Garante que o shapefile está em um sistema de coordenadas projetado (metros)
    if gdf.crs and gdf.crs.is_geographic:
        print("Convertendo de coordenadas geográficas para projetadas (UTM)...")
        # Determina a zona UTM apropriada baseada no centroide do shapefile
        centroid = gdf.unary_union.centroid
        utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
        hemisphere = 'south' if centroid.y < 0 else 'north'
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        gdf = gdf.to_crs(utm_crs)
    
    # Obtém os limites do shapefile
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Calcula o número de pixels necessários
    width = int((maxx - minx) / pixel_size) + 1
    height = int((maxy - miny) / pixel_size) + 1
    
    # Cria a transformação para o raster
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Cria um array vazio para o raster
    raster_data = np.zeros((height, width), dtype=np.uint8)
    
    # Rasteriza o shapefile
    shapes = [(geom, 1) for geom in gdf.geometry]
    burned = features.rasterize(
        shapes=shapes,
        out=raster_data,
        transform=transform,
        fill=0,
        all_touched=True
    )
    
    # Cria o perfil do raster
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.uint8,
        'crs': gdf.crs,
        'transform': transform,
        'nodata': 0
    }
    
    # Salva o raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(burned, 1)
    
    print(f"Raster criado com sucesso em: {output_path}")
    print(f"Resolução: {pixel_size}m x {pixel_size}m")
    print(f"Dimensões: {width} x {height} pixels")
    
    return output_path

def sum_raster_pair(df_rasters: DataFrame) -> np:
    list_rasters = list(df_rasters["path"])

    rasters_acumulados_hora = []
    rasters_acumulados_par_2h = []

    for first, second in zip(list_rasters[0::2], list_rasters[1::2]):
        rasters_acumulados_hora.append(
            np.asarray(accumulate_bands([first, second], band=1))
        )

    for first, second in zip(rasters_acumulados_hora, rasters_acumulados_hora[1:]):
        stack = np.stack([first, second])
        stack_sum = np.nansum(stack, 0)
        rasters_acumulados_par_2h.append(stack_sum)

    stack_24h_pair = np.stack(rasters_acumulados_par_2h)
    np_stack = np.asarray(stack_24h_pair)
    max_from_stack = np.amax(np_stack, axis=0)

    return max_from_stack


#@logger_wraps(level="DEBUG")
def accumulate_bands(paths: list, band: int = 1):
    """Function to accumulate a list of rasters, open all rasters
    and read band. Using the dask to stack all rasters bands. Return
    the accumulate (sum of rasters), sum with dask.

    Parameters
    ----------
    paths : list
        A list of paths for rasters
    band : int, optional
        The number of the band, by default 1

    Returns
    -------
    dask Array
        A array of the sum between all rasters
    """
    map2array = []
    for p in paths:
        map2array.append(rasterio.open(p, num_threads="all_cpus").read(band))

    ds_stack = da.stack(map2array)

    return da.nansum(ds_stack, 0)




#@logger_wraps(level="DEBUG")
def process_file(path, band, window, reference_width, reference_height):
    with rasterio.open(path) as src:
        if src.meta['width'] == reference_width and src.meta['height'] == reference_height:
            data = src.read(band, window=window, masked=True)
            data = np.where(data.mask, 0, data.data)
            return data
        else:
            return None
        
def accumulate_bands_parallel(paths, band, window, reference_width, reference_height):
    with Pool() as pool:
        results = pool.map(partial(process_file, band=band, window=window, 
                                   reference_width=reference_width, 
                                   reference_height=reference_height), paths)
    
    results_without_none = [item for item in results if item is not None]
    summed = np.sum(results_without_none, axis=0)
    return np.maximum(summed, 0)

def sum_to_file(paths: list, outfilename: str, overwrite_meta: dict, band: int = 1):
    """Accumulated a list of rasters to a tif file (sum all rasters of
    the band).

    Parameters
    ----------
    paths : list
        A list of paths for rasters
    outfilename : str
        The filename of the output file
    overwrite_meta : dict
        the profile of the raster (metadata) that need to be overwrite
    band : int, optional
        The number of the band, by default 1
    """
    with rasterio.open(paths[0]) as src0:
        meta = src0.meta.copy()
        windows = [window for ij, window in src0.block_windows()]
        reference_width = src0.meta['width']
        reference_height = src0.meta['height']

    meta.update(overwrite_meta)

    with rasterio.open(outfilename, "w", **meta) as out:
        for window in windows:
            o = accumulate_bands_parallel(paths, band, window, reference_width, reference_height)
            out.write(o, indexes=band, window=window)


# @logger_wraps(level="DEBUG")
def clip_to_file(
    infilename: str, outfilename: str, pol: Polygon, overwrite_meta: dict = {}
):
    """From a input file, generate a clipped raster from function
    clip(). Save the into the output raster file the clipped data.

    Parameters
    ----------
    infilename : str
        The filename of the input file
    outfilename : str
        The filename of the output file
    pol : Polygon
        Polygon with the georeferenced coordinates
    overwrite_meta : dict, optional
        the profile of the raster (metadata) that need to be overwrite,
        by default {}
    """

    with rasterio.open(infilename, "r") as input_raster:
        with clip(input_raster, pol) as clipped:
            meta = clipped.meta
            meta.update(overwrite_meta)
            with rasterio.open(outfilename, "w+", **meta) as out:
                out.write_band(1, clipped.read(1))


def xy_to_pol(r, x, y):
    ul = r.xy(x, y, offset="ul")
    ur = r.xy(x, y, offset="ur")
    ll = r.xy(x, y, offset="ll")
    lr = r.xy(x, y, offset="lr")
    return Polygon([ul, ur, lr, ll])


def get_data(path):

    r = path.split("/")
    year = r[-4]
    month = r[-3]
    day = r[-2]
    date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")

    return date


def read_precipitation_chunk(hdf, start_row, chunk_size, date_file):

    if date_file < datetime(2024, 6, 1, 0, 0, 0):
        chunk = hdf["Grid/precipitationCal"][0, start_row : start_row + chunk_size, :]
    else:
        chunk = hdf["Grid/precipitation"][0, start_row : start_row + chunk_size, :]
    return np.flipud(chunk)


def process_chunk(
    chunk: np.ndarray,
    transform: rasterio.Affine,
    template_raster: rasterio.io.DatasetReader,
    extent: Polygon,
    outfile: str,
    y_offset: int,
) -> None:
    meta = template_raster.meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "dtype": rasterio.float64,
            "height": chunk.shape[0],
            "width": chunk.shape[1],
            "crs": template_raster.crs,
            "transform": rasterio.Affine(
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f + y_offset * transform.e,
            ),
            "nodata": template_raster.nodata,
            "compress": None,
        }
    )

    with rasterio.open(outfile, "r+") as dst:
        dst.write(
            chunk,
            1,
            window=rasterio.windows.Window(
                col_off=0, row_off=y_offset, width=chunk.shape[1], height=chunk.shape[0]
            ),
        )

def __imerg_hdf5_to_gpm_raster__(
        path: str,
        outfile: str,
        extent: Polygon,
        template_raster: Raster) -> None:
    """This method read a HDF5 file from IMERG data (RAW from GPM),
     raw file at <path>, transform into a raster with rasterio, clip
     the raster to the <extent> bounds, extract the values of the
     RAW data from the imerg to the wrf centroides pixels points
     (<points_wrf>), and using these values with the
     <template raster>, write a new raster with the values of GPM
     data in the grid of the WRF data at with the name of <outfile>.

     Parameters
     ----------
     path : str
         Path for the GPM raw data.
     outfile : str
        Path for the output data, tif (grid of WRF), with values of
        GPM data.
     extent : Polygon
         Polygon with the bounds of the extent of the ROI
     template_raster : Raster
         A rasterio raster with the WRF grid (transform) and metadata
     """

    logger.info("opening HDF5 file...")
    with h5py.File(path, "r") as hdf:
        logger.info("Reading precipitation values...")
        date_file = get_data(path)
        # if date_file < datetime(2024, 6, 1, 0, 0, 0):
        #     precipitation_values = hdf["Grid/precipitationCal"][:]
        # else:
        precipitation_values = hdf["Grid/precipitation"][:]
        precipitation_values = np.flipud(precipitation_values)

        logger.info("Loading the lat and long values...")
        lats = hdf["Grid/lat"][:]
        longs = hdf["Grid/lon"][:]

        logger.info("Defining the transform for the raster...")
        transform = rasterio.transform.from_origin(
            longs[0], lats[-1], float(lats[1] - lats[0]), float(longs[1] - longs[0])
        )

        logger.info("Creating the raster file...")

        meta = template_raster.meta.copy()
        meta.update({"driver": "GTiff", "dtype": rasterio.float64, "height": precipitation_values.shape[1],
                     "width": precipitation_values.shape[2], "crs": template_raster.crs, "transform": transform,
                     "nodata": template_raster.nodata, "compress": None})

        with rasterio.open(outfile, "w+", **meta) as dst:
            dst.write(precipitation_values[0], 1)

        template_bounds = template_raster.bounds
        extent = box(*template_bounds)

        logger.info("Clipping the raster to the extent bounds...")
        with rasterio.open(outfile, "r") as src:
            out_image, out_transform = mask(src, [mapping(extent)], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                             "transform": out_transform, "compress": None})

        with rasterio.open(outfile, "w", **out_meta) as dst:
            dst.write(out_image[0], 1)


def open_raster(dir, prof):
    """
    Open and read multiple raster files from a directory.

    This function reads all the raster files located in the specified directory,
    and returns a list of tuples, each containing the file name, raster values,
    and a provided profile for the rasters.

    Parameters:
    ------------
    dir: str
        The directory path where the raster files are located.
    prof: dict
        A dictionary representing the raster profile to be used for
        the opened rasters.

    Returns:
    ----------
    raster_list: list of tuples:
        A list of tuples, where each tuple consists of the following:
            - file (str): The name of the raster file.
            - rst_values (numpy.ndarray): An array containing the raster values.
            - prof (dict): The provided or updated raster profile.
    """
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    raster_list = []
    for file in files:
        raster_path = os.path.join(dir, file)
        rst = rasterio.open(raster_path)
        rst_values = rst.read(1)
        prof.update(dtype=rasterio.int8, nodata=99, count=1)
        raster_list.append((file, rst_values, prof))

    return raster_list

def reproject_raster_and_clip(file, output_file, polygon):
    """
    Reproject a raster file to a specified CRS and clip it to a polygon geometry.

    Paramns:
    -------
    file: str
        Path to the input raster file.
    output_file: str
        Path to the output raster file.
    polygon: shapely.Object
        The polygon geometry to which the raster will be clipped.
    """
    with rasterio.open(file) as scl:
        logger.info("Starting reprojecting and cropping...")
        # Calculate the transform to the desired CRS
        transform, width, height = calculate_default_transform(
            scl.crs, "EPSG:4674", scl.width, scl.height, *scl.bounds
        )
        kwargs = scl.meta.copy()
        kwargs.update(
            {
                "driver": "GTiff",
                "crs": "EPSG:4674",
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        # Reproject the raster data
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**kwargs) as destination:
                reproject(
                    source=rasterio.band(scl, 1),
                    destination=rasterio.band(destination, 1),
                    src_transform=scl.transform,
                    dst_transform=transform,
                    src_crs=scl.crs,
                    dst_crs="EPSG:4674",
                    resampling=Resampling.nearest,
                )
                with clip(destination, polygon) as clipped:
                    meta = clipped.meta
                    with rasterio.open(output_file, "w+", **meta) as out:
                        out.write_band(1, clipped.read(1))
                        logger.success("Reproject and crop finished successfully!")


def resample_band(band, xres, yres, output_file, polygon):
    """
    Resample a band to a specified resolution and clip it to a polygon geometry.

    Args:
    band: str
        Path to the input band file.
    xres: float
        Target resolution along the x-axis.
    yres: float
        Target resolution along the y-axis.
    output_file: str
        Path to the output file.
    polygon: shapely.Object
        The polygon geometry to which the raster will be clipped.
    """
    logger.info(f"Resampling 20m resoltion to {xres}...")
    with rasterio.open(band) as dataset:
        scale_factor_x = dataset.res[0] / xres
        scale_factor_y = dataset.res[1] / yres

        profile = dataset.profile.copy()
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor_y),
                int(dataset.width * scale_factor_x),
            ),
            resampling=Resampling.nearest,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x), (1 / scale_factor_y)
        )

        profile.update(
            {
                "height": data.shape[-2],
                "width": data.shape[-1],
                "transform": transform,
                "driver": "GTiff",
            }
        )

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)

            logger.success(f"Resampling complete successfully!")
            reproject_raster_and_clip(memfile.name, output_file, polygon)
            logger.success(
                "Resampling, reprojecting and cropping complete successfully!"
            )


def stacking_sentinel_bands(paths, output_path, mask):
    """
    Stack multiple Sentinel bands and save the stacked bands to a GeoTIFF file.

    Params:
    --------
    paths: list
        List of paths to Sentinel bands.
    output_path: str
        Path to the output GeoTIFF file.
    mask: numpy.ndarray
        Mask to apply to each band.
    """
    bands_data = []
    for path in paths:
        with rasterio.open(path) as src:
            band_data = src.read(1)  # Assuming bands are 1-indexed
            if band_data.shape[1] > mask.shape[1]:
                band_data = band_data[:, : mask.shape[1]]
            elif band_data.shape[1] < mask.shape[1]:
                band_data = np.pad(
                    band_data,
                    ((0, 0), (0, mask.shape[1] - band_data.shape[1])),
                    mode="constant",
                    constant_values=0,
                )

            if band_data.shape[0] > mask.shape[0]:
                band_data = band_data[: mask.shape[0], :]
            elif band_data.shape[0] < mask.shape[0]:
                band_data = np.pad(
                    band_data,
                    ((0, mask.shape[0] - band_data.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            masked_band_data = band_data * mask

            bands_data.append(masked_band_data)

    # Stack the bands
    stacked_data = np.dstack(bands_data)

    # Get the metadata from one of the input files
    with rasterio.open(paths[0]) as src:
        profile = src.profile

    # Update the metadata for the stacked data
    profile.update(
        {
            "count": stacked_data.shape[2],
            "dtype": stacked_data.dtype,
            "nodata": 0,
            "compression": "lzw",
        }
    )

    # Write the stacked data to a GeoTIFF file
    with rasterio.open(output_path, "w", **profile) as dst:
        # Write each band of the array
        for i in range(stacked_data.shape[2]):
            dst.write(stacked_data[:, :, i], i + 1)


def mask_scl(scl_path):
    """
    Create a mask from the Scene Classification Layer (SCL) raster data.

    Params:
    --------
    scl_path: str
        Path to the SCL raster file.

    Returns:
    ---------
    mask: numpy.ndarray
        Mask array.
    profile: dict
        Metadata of the source file.
    """
    with rasterio.open(scl_path) as src:
        # Read the raster data
        raster_data = src.read(1)

        # Define the values to include in the mask
        included_values = [0, 1, 2, 3, 8, 9, 10, 11]

        # Create the mask
        mask = np.isin(raster_data, included_values)
        mask = ~mask

        # Get the metadata from the source file
        profile = src.profile

    # Set the count to 1 as we have a single band
    profile["count"] = 1

    # Update the dtype to uint8
    profile["dtype"] = "uint8"

    # Update the nodata value
    profile.pop("nodata", None)

    return mask, profile


def classifying_land_use(stacked_raster_path, modelo, output_filename):
    with rasterio.open(stacked_raster_path) as src:
        raster_stack = src.read()
        transform = src.transform

        # Reformatar o raster para que cada coluna contenha os valores de uma banda
        raster_reshaped = np.reshape(raster_stack, (raster_stack.shape[0], -1)).T
        valid_indices = np.all(raster_reshaped != 0, axis=1)
        raster_filtered = raster_reshaped[valid_indices]

        # Aplicar o modelo apenas nos dados filtrados
        classificacoes = modelo.predict(raster_filtered)
        classificacoes = classificacoes.astype(np.uint16)

        # Criar um array de classificações com o mesmo tamanho do raster original
        classificacoes_full = np.zeros(
            (src.height * src.width,), dtype=classificacoes.dtype
        )
        classificacoes_full[valid_indices] = classificacoes
        classificacoes_full = classificacoes_full.reshape((src.height, src.width))

        with rasterio.open(
            output_filename,
            "w",
            driver="GTiff",
            width=classificacoes_full.shape[1],
            height=classificacoes_full.shape[0],
            count=1,
            dtype=classificacoes_full.dtype,
            crs=src.crs,
            transform=transform,
        ) as dst:
            dst.write(classificacoes_full, 1)


def round_point_coordinates(point, decimals):
    return Point(round(point.x, decimals), round(point.y, decimals))


def raster_to_gdf(raster_files):
    gdf_list = []
    for raster in raster_files:
        dataarray = rxr.open_rasterio(raster)
        df = (
            dataarray[0].to_pandas().stack().reset_index(name="classificacao_uso_terra")
        )
        df = df[df["classificacao_uso_terra"] != 0]
        df[["x", "y"]] = df[["x", "y"]].round(6)
        geometry = gpd.points_from_xy(df["x"], df["y"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=dataarray.rio.crs)

        gdf_list.append(gdf)

    return gdf_list


def create_mosaico(bands, output_filename):
    # Lista para armazenar os datasets das bandas
    datasets = []
    # Abrir cada banda e adicionar ao datasetu
    for band in bands:
        datasets.append(rasterio.open(band))
    crs = datasets[0].crs
    # Combine os datasets em um único mosaico
    mosaic, transform = merge(datasets, resampling=Resampling.nearest)

    with rasterio.open(
        output_filename,
        "w",
        driver="JP2OpenJPEG",
        count=mosaic.shape[0],
        dtype=mosaic.dtype,
        crs=crs,
        transform=transform,
        width=mosaic.shape[2],
        height=mosaic.shape[1],
    ) as dst:
        dst.write(mosaic)


def align_rasters(raster_ref, raster_to_align, output_filename):
    # Abrir os rasters
    raster1 = gdal.Open(raster_ref)
    raster2 = gdal.Open(raster_to_align)

    # Resample raster2 para que tenha o mesmo tamanho de pixel que raster1, salvando em memória
    raster2_resampled_mem = gdal.Warp(
        "",
        raster2,
        xRes=raster1.GetGeoTransform()[1],
        yRes=raster1.GetGeoTransform()[5],
        format="MEM",
    )

    # Obter informações sobre as transformações geométricas
    geotransform_raster1 = raster1.GetGeoTransform()
    geotransform_raster2_resampled = raster2_resampled_mem.GetGeoTransform()

    # Definir as transformações geométricas para o raster2 resampleado para alinhar com o raster1
    new_geotransform = (
        geotransform_raster1[0],  # origem X
        geotransform_raster1[1],  # tamanho de pixel X (mesmo que raster1)
        geotransform_raster2_resampled[2],  # inclinação X (rotação, geralmente 0)
        geotransform_raster1[3],  # origem Y
        geotransform_raster2_resampled[4],  # inclinação Y (rotação, geralmente 0)
        geotransform_raster1[5],  # tamanho de pixel Y (mesmo que raster1)
    )

    # Definir a resolução de destino
    target_resolution = [geotransform_raster1[1], geotransform_raster1[5]]

    # Aplicar a transformação geométrica ao raster2 resampleado
    raster2_aligned_mem = gdal.Warp(
        "",
        raster2_resampled_mem,
        outputBounds=(
            geotransform_raster1[0],
            geotransform_raster1[3],
            geotransform_raster1[0] + raster1.RasterXSize * geotransform_raster1[1],
            geotransform_raster1[3] + raster1.RasterYSize * geotransform_raster1[5],
        ),
        outputBoundsSRS="EPSG:4674",
        xRes=target_resolution[0],
        yRes=target_resolution[1],
        dstSRS="EPSG:4674",
        outputType=gdal.GDT_UInt16,
        format="MEM",
    )

    # Aplicar a compactação LZW e salvar o resultado final
    gdal.Translate(
        output_filename, raster2_aligned_mem, creationOptions=["COMPRESS=LZW"]
    )

    # Fechar os rasters
    raster1 = None
    raster2 = None
    raster2_resampled_mem = None
    raster2_aligned_mem = None


def raster_mapbiomas_to_gdf(raster_file):
    dataarray = rxr.open_rasterio(raster_file)
    df = dataarray[0].to_pandas().stack().reset_index(name="classificacao_uso_terra")
    df = df[df["classificacao_uso_terra"].isin([1, 2, 3, 4, 5, 6])]
    df[["x", "y"]] = df[["x", "y"]].round(6)
    geometry = gpd.points_from_xy(df["x"], df["y"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=dataarray.rio.crs)
    return gdf   

def load_shapefile(filepath):
    gdf = gpd.read_file(filepath)
    return gdf  