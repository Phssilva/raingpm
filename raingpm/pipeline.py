import calendar

from datetime import datetime, timedelta, timezone
from src.DownloadGPMData import DownloadGPM
from src.PrepareGPMData import PrepareGPM
from src.ProcessingGPMData import ProcessingGPM

from loguru import logger


def get_last_day_of_month(year: int) -> datetime:
    """Get the last day of the month"""

    last_days = []
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        last_day_datetime = datetime(year, month, last_day, 23, 59)

        last_days.append((last_day_datetime, last_day))

    return last_days


def data_ingestion(date: tuple):
    """Function responsible for downloading GPM data"""
    gpm = DownloadGPM()
    return gpm.download_gpm_data(date)


def data_prepare(date: tuple):
    """Function responsible for preparing GPM data"""
    gpm = PrepareGPM()
    gpm.prepare_gpm_data(date)


def data_processing(date: tuple):
    """Function responsible for processing GPM data"""
    gpm = ProcessingGPM()
    gpm.processing_gpm_data(date)


if __name__ == "__main__":
    dt_end = datetime(2024,1,18)
    dt_end = dt_end - timedelta(hours=3)
    dt_init = dt_end - timedelta(days=7)
    # data_ingestion(date=(dt_end, dt_init))
    # last_days_of_month = get_last_day_of_month(2024)

    # for date in last_days_of_month:
    date = (dt_init, dt_end)
    logger.info(f'Processing data from {date[0]} to {date[1]}')
    r = data_ingestion(date)
    if r:
        data_prepare(date)
        data_processing(date)
