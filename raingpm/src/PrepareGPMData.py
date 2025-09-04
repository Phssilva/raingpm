import os
import rasterio


import pandas as pd
import geopandas as gpd


from datetime import datetime, timedelta
from loguru import logger
from config import settings
from misc.files import Files
from misc.geo import raster


class PrepareGPM():
    """Class responsible for realize prepare GPM data
    """
    
    def __init__(self) -> None:
        self.data_types = ['EARLY','LATE']
        self.dist_dir = settings.STORAGE.gpm_intermediary
    
    def _get_path_for_prepare(self):
        """Get the path of the result of the prepare GPM data
        """
        path_result = self.dist_dir
        path_result = os.path.join(path_result, 'rain_hourly/EARLY')
        
        return path_result
        
    def _get_path_download(self):
        """Get the path of the download GPM data
        """
        path_result = settings.STORAGE.gpm_raw
        
        return path_result
    
    def get_polygon_from_template(self):
        """Get the polygon from template
        """
        
        gdf = gpd.read_file(settings.STORAGE.extent_gpm_template)
        
        polygon = gdf.geometry.iloc[0]
        
        return polygon
    
    def get_roi_from_hdf5(self, df_files_downloaded, dst_dir):
        
        files = Files()
        raster_template = settings.STORAGE.raster_gpm_template
        
        df_files_downloaded = pd.DataFrame(df_files_downloaded,
                                           columns=['path',
                                                    'date',
                                                    'type'])
        
        date_files = df_files_downloaded['date'].unique().tolist()
        date_files = pd.DataFrame(date_files, columns=['date'])
        
        df_files_downloaded['date'] = pd.to_datetime(df_files_downloaded['date'])
        
        df_files = files.generate_directory_prepare(df_files_downloaded,
                                                    dst_dir,
                                                    'rain_hourly')
        
        logger.info(f'Cropping {df_files.shape[0]} files...')
        
        extent = self.get_polygon_from_template()
        template_raster = rasterio.open(raster_template)
        
        for it, row in df_files.iterrows():
             logger.info(f'Cropping {row["path"]}...')
             filename_dst = files.gen_hourly_name(row['out_dir'], row['path'])
             raster.__imerg_hdf5_to_gpm_raster__(row['path'], filename_dst, extent, template_raster)


    def _prepare_gpm_data(self, date:tuple):
        
        files = Files()
        path_prepare = self._get_path_for_prepare()
        path_download = self._get_path_download()
        date_init = date[0]
        date_end = date[1]
        
        logger.info(f'Preparing GPM data from {date_init} to {date_end}')
        
        df_files_downloaded = files.search_files(date_init,
                                                date_end,
                                                path_download,
                                                self.data_types[0],
                                                "raw"
                                                )
        
        logger.info(f'Found {len(df_files_downloaded)} files to prepare')
        
        self.get_roi_from_hdf5(df_files_downloaded, path_prepare)

    def prepare_gpm_data(self, date:tuple):
        """Prepare the GPM data
        """
        self._prepare_gpm_data(date)