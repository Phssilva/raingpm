""" Module responsible for processing GPM data """

import os

import pandas as pd

from config import settings
from misc.files import Files
from misc.geo import raster


class ProcessingGPM:
    """Class responsible for realize prepare GPM data"""

    def __init__(self) -> None:
        self.data_types = ["EARLY", "LATE"]
        self.origin_dir = settings.STORAGE.gpm_prepared
        self.dst_dir = settings.STORAGE.gpm_processed

    def processing_gpm_data(self, date: tuple):
        """Function responsible for processing GPM data"""

        files = Files()

        intermediary_files = files.search_files(
            date[0], date[1], self.origin_dir, "EARLY"
        )

        df_intermediary_files = pd.DataFrame(
            intermediary_files, columns=["path", "file_datetime", "type"]
        )

        category = df_intermediary_files["file_datetime"].max().strftime("%b")
        year = df_intermediary_files["file_datetime"].max().strftime("%Y")
        dst_dir = os.path.join(self.dst_dir, year, category)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        raster.sum_rasters(df_intermediary_files, category, dst_dir)
        print(df_intermediary_files)
