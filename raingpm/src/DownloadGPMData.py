import tempfile
import os
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from loguru import logger
from config import settings
from misc.files import Files


class DownloadGPM:
    """Class responsible for downloading GPM data."""

    ACCEPT_TYPES = ['.RT-H5']

    def __init__(self, period: tuple = (datetime.now(), datetime.now())):
        """Initialize with destination directory and period."""
        self.dst_dir = settings.STORAGE.gpm_raw
        self.period = period
        self.nmax_process = settings.DEFAULT.nmax_process
        self.URL_BASE = settings.GPM.url
        self.CRD_EMAIL = settings.GPM.user
        self.CRD_PASWR = settings.sec.GPM.password

    @property
    def urls(self) -> dict:
        """Returns URLs for EARLY and LATE data."""
        return {
            "EARLY": f"{self.URL_BASE}early/",
            "LATE": f"{self.URL_BASE}late/"
        }

    def get_max_processes(self, max_process: int = 8) -> int:
        """Determine the number of processes to use."""
        return min(cpu_count(), max_process)

    def filter_files_by_period(self, filenames: list) -> list:
        """Filter files based on the specified period."""
        files = Files()
        return files.remove_by_period(filenames, self.period)

    def get_file_datetime(self, filename: str, is_start: bool = True) -> datetime:
        """Extract datetime from filename."""
        period = os.path.basename(filename).split('.')[-3].split('-')
        date_str = f"{period[0]}_{period[1 if is_start else 2]}"
        return datetime.strptime(date_str, "%Y%m%d_S%H%M%S")

    def create_directory(self, date: datetime, type_data: str) -> str:
        """Create destination directory for downloads."""
        files = Files()
        date_path = files.datetime_to_path(date)
        dst = os.path.join(self.dst_dir, type_data, date_path)
        os.makedirs(dst, exist_ok=True)
        return dst

    def download_files(self, urls: list, type_data: str, batch: int = 250) -> None:
        """Download files in parallel."""
        logger.info("Setting up temporary directory...")
        temp_dir = tempfile.TemporaryDirectory()

        process_count = self.get_max_processes(self.nmax_process)
        logger.info(f"Downloading {len(urls)} files with {process_count} processes...")
        pool = Pool(processes=process_count)
        download_func = partial(Files().download_file, dst=self.dst_dir, dst_tmp=temp_dir.name)

        for i, _ in enumerate(pool.imap_unordered(download_func, urls)):
            if i % batch == 0 and i > 0:
                logger.info(f"{i}/{len(urls)} files downloaded...")

        pool.close()
        pool.join()
        logger.info(f"Download complete. Cleaning up temporary directory.")
        temp_dir.cleanup()

    def download_by_date(self, date_range: tuple, type_data: str = "EARLY", batch: int = 250) -> str:
        """Download all files for the specified date range and type."""
        self.period = date_range
        files = Files()

        url = self.urls[type_data]
        date_str = date_range[1].strftime("%Y%m")
        url_complet = f"{url}{date_str}/"

        logger.info("Fetching file list...")
        filenames = files.get_filenames(url_complet, self.ACCEPT_TYPES)
        filenames_to_download = self.filter_files_by_period(filenames)

        if filenames_to_download:
            download_urls = files.mount_url_for_download(url_complet, filenames_to_download)
            self.download_files(download_urls, type_data, batch)
            logger.info(f"Downloaded {len(download_urls)} files from {url_complet}")
            return True
        else:
            logger.info(f"No files to download from {url_complet}")
            return False

    def download_gpm_data(self, date_range: tuple) -> str:
        """Main method to download GPM data."""
        return self.download_by_date(date_range, type_data="EARLY")