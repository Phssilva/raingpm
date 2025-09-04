import os
import requests
import re
import shutil

from typing import List
from loguru import logger
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup as bs

from config import settings


class Files:
    """Class to manage files and directories"""

    def __init__(self) -> None:
        pass

    def gen_hourly_name(self, dir: str, file_path: str) -> str:
        """Generate the hourly name of tif files.
            Joins the dir, with a dir by the data_type and sub dirs
            based on file_datetime (%Y/%m/%d/).
        Parameters
        ----------
        dir : str
            The basedir to generate the full filename.
        subdir : str
            The sub dir after the dates dir to generate the
            full filename.
        file_path : str
            the path of the raw file
        data_type : str
            Data type of the raw data (EARLY or LATE)
        file_datetime : datetime
            The datetime of the file reference to compute the hourly
            rain.

        Returns
        -------
        str
            A full filename of the tif files hourly.
        """
        bn = self.get_basename(file_path)
        bn = bn.split(".")[-3].split("-")
        start_dt = datetime.strptime(f"{bn[0]}_{bn[1]}", "%Y%m%d_S%H%M%S")
        start_dt = datetime.strftime(start_dt, "S%Y%m%d-%H%M%S")
        end_dt = datetime.strptime(f"{bn[0]}_{bn[2]}", "%Y%m%d_E%H%M%S")
        end_dt = datetime.strftime(end_dt, "E%Y%m%d-%H%M%S")
        filename = f"{start_dt}_{end_dt}.tif"
        dir_out = os.path.join(dir, filename)

        return dir_out

    def get_extension(self, filename: str):
        """Convert a full filename for just the the extension

        Parameters
        ----------
        str : filename
        full filename of the file

        Returns
        -------
        str
            the file extension
        """
        return os.path.splitext(os.path.basename(filename))[1]

    def datetime_to_path(self, dt: datetime, subdir: str = "") -> str:
        """Transform a datetime to a group of directories

        Parameters
        ----------
        dt : datetime
            the datetime of dessired to transform into a directory
        subdir : str, optional
            A sub directory {suffix} to concat at out, by default ''

        Returns
        -------
        str
            Path for a group of directories
        """
        dir = datetime.strftime(dt, f"%Y/%m/%d/{subdir}")
        return dir

    def search_files(
        self,
        start_date: datetime,
        end_date: datetime,
        path: str,
        type_path: str,
        path_storage: str = "intermediary",
    ):
        """Search files in the download directory"""

        _files = []

        dates_to_search = [
            end_date - timedelta(days=x)
            for x in range((end_date - start_date).days + 1)
        ]
        folders_to_search = [
            os.path.join(path, date.strftime("%Y/%m/%d")) for date in dates_to_search
        ]

        for folder in folders_to_search:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    _files.append(file_path)
                    _files.append(self.get_file_datetime(filename, type_path= path_storage))
                    _files.append(type_path)

        _files = [_files[i : i + 3] for i in range(0, len(_files), 3)]
        
        # Garantir que as datas estejam no mesmo formato (com ou sem timezone)
        start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
        
        _files = [
            file for file in _files if file[1] >= start_date_naive and file[1] <= end_date_naive
        ]
        _files.sort(key=lambda x: x[1])

        return _files

    def get_filenames(self, url: str, accept_types: List[str]) -> list:
        """Get all filenames to download
        Parameters
        ----------
        url : str
        URL of the imerg data (nasa) with type and date
        Returns
        -------
        list
            list with only the filenames available to download
        """
        logger.info(f"Logging on {url}")
        page = requests.get(url, auth=(settings.GPM.user, settings.sec.GPM.password))

        logger.info("Scrapping page...")
        soup = bs(page.text, "html.parser")
        html_td = bs(str(soup.findAll("td")), "html.parser")
        html_a = html_td.findAll("a")
        filenames = [fn.get("href") for fn in html_a]
        o = [x for x in filenames if self.get_extension(x) in accept_types]

        logger.info(
            f"The scrapping results in {len(o)} files with extension {accept_types}"
        )

        return o

    def download_file(self, url: str, dst: str, dst_tmp: str):
        """download file from a url
        Downloads the file from a link, is ignored if the file is
        already downloaded

        Parameters
        ----------
        url : str
            URL of the imerg data (nasa)
        dst : str
            path of the dir to save the files
        dst_tmp : list
            path of the temporary directory to save the files
        -------
        Returns
            str
                Success or error message
        """
        filename = url.split("/")[-1]

        dt = self.get_file_datetime(filename, type_path="raw")
        dst_dt = os.path.join(dst, self.datetime_to_path(dt))
        os.makedirs(dst_dt, exist_ok=True)

        try:
            # Usar streaming para garantir download completo
            response = requests.get(
                url, stream=True, auth=(settings.GPM.user, settings.sec.GPM.password),
                timeout=60  # Adicionar timeout para evitar downloads infinitos
            )
            
            # Verificar se a resposta foi bem-sucedida
            response.raise_for_status()
            
            filename_tmp = os.path.join(dst_tmp, filename)
            
            # Download em chunks para arquivos grandes
            with open(filename_tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filtrar keep-alive chunks vazios
                        f.write(chunk)
            
            # Verificar se o arquivo é um HDF5 válido antes de mover
            try:
                # Tentar abrir o arquivo para verificar se é um HDF5 válido
                import h5py
                with h5py.File(filename_tmp, 'r', driver='core') as _:
                    pass  # Se abrir sem erro, o arquivo é válido
                
                # Arquivo válido, mover para destino final
                out_fn = os.path.join(dst_dt, filename)
                shutil.copyfile(filename_tmp, out_fn)
                os.remove(filename_tmp)
                
                return (
                    out_fn,
                    {
                        "state": "success",
                        "url": url,
                        "creation_datetime": datetime.now(),
                        "file_datetime": dt,
                    },
                )
            except Exception as e:
                # Arquivo HDF5 inválido
                logger.error(f"Arquivo baixado não é um HDF5 válido: {e}")
                if os.path.exists(filename_tmp):
                    os.remove(filename_tmp)
                return (filename, {"state": "fail", "error": f"Arquivo HDF5 inválido: {e}"})
                
        except Exception as e:
            logger.warning(f"Falha ao baixar arquivo {filename}. Erro -> {e}")
            return (filename, {"state": "fail", "error": e})

    def remove_by_period(self, filenames: list, period: tuple) -> list:
        """From a list of filenames, select those that are between the
        time period defined at context.

        Parameters
        ----------
        filenames : list
            A list of strings that contains the filenames

        Returns
        -------
        list
            A list of filenames in the period of time desired.
        """
        logger.info("Selecting files just in the period...")

        # Ensure the period is naive if get_file_datetime is naive
        period_naive = (period[0].replace(tzinfo=None), period[1].replace(tzinfo=None))

        i__ = {
            fn: {
                "start": self.get_file_datetime(fn, type_path="raw"),
                "end": self.get_file_datetime(fn, False, type_path="raw"),
            }
            for fn in filenames
        }

        return [
            fn
            for fn, v in i__.items()
            if v["start"] >= period_naive[0] and v["end"] <= period_naive[1]
        ]

    def generate_directory_prepare(
        self, df_files, dir_out, etc: str = "rain_hourly"
    ) -> None:
        for file in df_files.iterrows():
            out_dir = f"{dir_out}/{self.datetime_to_path(file[1][1])}"
            df_files.loc[file[0], "out_dir"] = out_dir
            os.makedirs(out_dir, exist_ok=True)
        return df_files

    def get_basename(self, filename: str, without_extesion: bool = True) -> str:
        """Convert a full filename for just the basename

        Parameters
        ----------
        filename : str
            The full path for the file
        without_extesion : bool, optional
            Determine if the basename that will return is with the extension
            or not, if False return basename with extension, by default True

        Returns
        -------
        str
            The basename of the filename
        """
        if without_extesion:
            return os.path.splitext(os.path.basename(filename))[0]
        else:
            return os.path.basename(filename)

    def mount_url_for_download(self, url_base: str, filenames: list) -> list:
        """Mount the url for download the file

        Parameters
        ----------
        filename : str
            The filename of the file
        url_base : str
            The base url of the file

        Returns
        -------
        list
            The urls for download the file
        """
        result = []
        for filename in filenames:
            result.append(f"{url_base}{filename}")
        return result

    def get_file_datetime(
        self, filename: str, s: bool = True, type_path: str = "intermediary"
    ) -> datetime:
        """Get the datetime from the filename, the GPM files have the
        start (reference time) and end datetime of the period of time
        that file represents. This method generate a datetime from
        these information.

        Parameters
        ----------
        filename : str
            The filename of a GPM file
        s : bool, optional
            If True the method return the start datetime, else
            will return the end datetime, by default True

        Returns
        -------
        datetime
            The datetime of start or end of the file
        """
        # if type_path == "intermediary":
        #     match = re.search(r"S(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", filename)
        #     return datetime.strptime(match.group(0), "S%Y%m%d-%H%M%S")

        bn = self.get_basename(filename)
        if type_path == "intermediary":
            match = re.search(r"S(\d{8}-\d{6})_E(\d{8}-\d{6})", bn)
            period = (
                match.group(1).replace("-", "_"),
                match.group(2).replace("-", "_"),
            )
            if s:
                return datetime.strptime(period[0], "%Y%m%d_%H%M%S")
            else:
                return datetime.strptime(period[1], "%Y%m%d_%H%M%S")
        else:
            period = bn.split(".")[-3].split("-")

        if s:
            return datetime.strptime(f"{period[0]}_{period[1]}", "%Y%m%d_S%H%M%S")
        else:
            return datetime.strptime(f"{period[0]}_{period[2]}", "%Y%m%d_E%H%M%S")
