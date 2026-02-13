import csv
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from google.cloud import bigquery, storage


class BuildMatrixParser:
    """Parses a build matrix CSV file to extract OE serials and their corresponding LHS and RHS serials."""

    def __init__(self, matrix_path: str) -> None:
        """Initialize the parser with the path to the build matrix CSV file."""

        """Args:
            matrix_path (str): Path to the build matrix CSV file.
        """

        """Returns:
            None
        """
        self.matrix_path = Path(matrix_path)

    def get_oe_to_lhs_rhs(self, oe_serials: list[str]) -> dict:
        """Fetches OE serials and their corresponding LHS and RHS serials from the build matrix.

        Args:
            oe_serials (list[str]): List of OE serials to look for in the build matrix.

        Returns:
            dict: A dictionary mapping OE serials to their corresponding LHS and RHS serials.
        """
        result = {}
        with self.matrix_path.open(newline="", encoding="utf-8") as csvfile:  # Updated this line
            reader = csv.DictReader(csvfile, delimiter=",")
            for row in reader:
                oe = row["OE serial"].strip()
                if oe in oe_serials:
                    lhs = row["LHS"].strip()
                    rhs = row["RHS"].strip()
                    result[oe] = (lhs, rhs)
        return result


class BigQueryFetcher:
    """Fetches test file URLs from Google BigQuery based on serial numbers."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize the BigQuery client.

        Args:
            project (str, optional): Google Cloud project ID. If not provided,
                will attempt to use the default from environment (GOOGLE_CLOUD_PROJECT).
        """
        self.client = bigquery.Client(project=project)

    def fetch_all_serials(self, station_filter: str | None = None, after_date: str | None = None) -> list[str]:
        """Fetches all unique serial numbers from BigQuery based on filters.

        Args:
            station_filter (str, optional): Filter to fetch serials only from a specific station.
            after_date (str, optional): Only fetch serials with data after this date (format: 'YYYY-MM-DD').

        Returns:
            list[str]: List of unique serial numbers matching the criteria.
        """
        query = """
        SELECT DISTINCT
            t_result.serialno AS serial
        FROM
            `labs-test-data.arrakis.test_data` AS t_result
        WHERE
            1=1
        """

        query_parameters = []

        if station_filter:
            query += " AND t_result.station = @station"
            query_parameters.append(bigquery.ScalarQueryParameter("station", "STRING", station_filter))

        if after_date:
            query += " AND t_result.date_time >= @after_date"
            query_parameters.append(bigquery.ScalarQueryParameter("after_date", "DATETIME", after_date))

        query += " ORDER BY serial"

        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        query_job = self.client.query(query, job_config=job_config)

        serials = [row["serial"] for row in query_job]
        print(f"Found {len(serials)} unique serial numbers matching the criteria.")
        return serials

    def fetch_test_file_urls(self, serial_numbers: list[str], after_date: str | None = None) -> dict[str, list[dict]]:
        """Fetches test file URLs from BigQuery for the given serial numbers.

        Args:
            serial_numbers (list[str]): List of serial numbers to fetch test file URLs for.
            after_date (str, optional): Only fetch files after this date (format: 'YYYY-MM-DD').

        Returns:
            dict: A dictionary mapping serial numbers to lists of test file URLs and metadata.
        """

        query = """
        SELECT
            t_result.serialno AS serial,
            t_result.station,
            t_result.date_time,
            t_files.url AS test_file_url,
            t_files.name AS test_file_name
        FROM
            `labs-test-data.arrakis.test_data` AS t_result, UNNEST(t_result.files) AS t_files
        WHERE
            t_result.serialno IN UNNEST(@serials)
        """

        if after_date:
            query += " AND t_result.date_time >= @after_date"

        query += """
        ORDER BY
            t_result.date_time DESC
        """

        query_parameters = [bigquery.ArrayQueryParameter("serials", "STRING", serial_numbers)]
        if after_date:
            query_parameters.append(bigquery.ScalarQueryParameter("after_date", "DATETIME", after_date))

        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)

        result: dict = {}
        query_job = self.client.query(query, job_config=job_config)

        for row in query_job:
            serial = row["serial"]
            result.setdefault(serial, []).append(
                {
                    "url": row["test_file_url"],
                    "file_name": row["test_file_name"],
                    "station": row["station"],
                    "date_time": row["date_time"],
                }
            )

        return result


class GCSDownloader:
    """Downloads files from Google Cloud Storage (GCS) based on URLs."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the GCS downloader.

        Args:
            output_dir (Path): Directory where files will be downloaded.
        """
        """Returns:
            None
        """
        self.output_dir = output_dir
        self.storage_client = storage.Client()

    def download(self, url: str, output_path: Path) -> None:
        """Downloads a file from GCS given its URL, skipping if the file already exists.

        Args:
            url (str): The GCS URL of the file to download.
            output_path (Path): The local path where the file will be saved.

        Returns:
            None
        """
        try:
            if output_path.exists():
                print(f"File already exists, skipping: {output_path}")
                return

            parsed_url = urlparse(url)
            bucket_name = parsed_url.path.split("/")[1]
            blob_name = parsed_url.path[len(bucket_name) + 2 :]
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(output_path))
            print(f"Downloaded {blob_name} to {output_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    def download_all(
        self,
        serial_to_files: dict,
        station_filter: str | None = None,
        file_filter: Callable[[str], bool] | None = None,
        limit_per_station: int | None = None,
    ) -> None:
        """Downloads all files from GCS based on the provided serial-to-files mapping.

        Args:
            serial_to_files (dict): A dictionary mapping serial numbers to lists of files (URLs and metadata).
            station_filter (str, optional): Filter to download files only from a specific station.
            file_filter (Callable[[str], bool], optional): A filter function to apply to the file paths.
            limit_per_station (int, optional): Maximum number of images to download per station.

        Returns:
            None
        """
        with ThreadPoolExecutor() as executor:
            futures = []
            station_download_count: dict[str, int] = {}

            for serial, files in serial_to_files.items():
                for file in files:
                    station = file["station"]
                    url = file["url"]
                    file_name = file["file_name"]

                    if station_filter and station != station_filter:
                        continue

                    # Fix applied here: Use file_name instead of rel_path
                    if file_filter and not file_filter(file_name):
                        continue

                    # Apply limit per station if specified
                    if limit_per_station is not None:
                        current_count = station_download_count.get(station, 0)
                        if current_count >= limit_per_station:
                            continue
                        station_download_count[station] = current_count + 1

                    destination = self.output_dir / station / serial / file_name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    futures.append(executor.submit(self.download, url, destination))

            for future in futures:
                future.result()


class TestFileManager:
    """Manages the process of fetching and downloading test files, optionally using a build matrix."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        parser: BuildMatrixParser,
        station_filter: str,
        image_filter: Callable[[str], bool],
        serial_list: list[str],
        limit_per_station: int | None = None,
        after_date: str | None = None,
        project: str | None = None,
    ) -> None:
        """Initialize the TestFileManager.

        Args:
            source_dir (str, optional): Directory containing the source OE folders.
            output_dir (str): Directory where files will be downloaded.
            parser (BuildMatrixParser, optional): An instance of BuildMatrixParser.
            station_filter (str, optional): Filter to download files only from a specific station.
            image_filter (Callable[[str], bool], optional): A filter function to apply to the file paths.
            serial_list (list[str], optional): List of OE serials to use instead of reading from folders.
            limit_per_station (int, optional): Maximum number of images to download per station.
            after_date (str, optional): Only download files after this date (format: 'YYYY-MM-DD').
            project (str, optional): Google Cloud project ID for BigQuery. If not provided,
                will attempt to use the default from environment (GOOGLE_CLOUD_PROJECT).
        """
        self.source_dir = source_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parser = parser
        self.station_filter = station_filter
        self.file_filter = image_filter
        self.serial_list = serial_list
        self.limit_per_station = limit_per_station
        self.after_date = after_date

        self.fetcher = BigQueryFetcher(project=project)
        self.downloader = GCSDownloader(self.output_dir)

    def get_oe_serials(self) -> list[str]:
        """Fetch OE serials either from a provided list, from directory names, or from BigQuery."""
        if self.serial_list is not None:
            print(f"Using provided serial list: {self.serial_list}")
            return self.serial_list
        if self.source_dir is not None:
            return [name for name in os.listdir(self.source_dir) if os.path.isdir(os.path.join(self.source_dir, name))]
        # Fetch all available serials from BigQuery based on filters
        print("No serial list provided. Fetching all available serials from BigQuery...")
        return self.fetcher.fetch_all_serials(station_filter=self.station_filter, after_date=self.after_date)

    def run(self) -> None:
        """Main method to run the test file manager."""
        oe_serials = self.get_oe_serials()
        print(f"Resolved OE serials: {oe_serials}")

        if self.parser:
            mapping = self.parser.get_oe_to_lhs_rhs(oe_serials)
            serials_to_fetch = sorted({s for pair in mapping.values() for s in pair})
            print(f"Using BuildMatrixParser. Mapped serials: {serials_to_fetch}")
        else:
            serials_to_fetch = oe_serials
            print(f"No BuildMatrixParser provided. Using OE serials directly: {serials_to_fetch}")

        serial_to_files = self.fetcher.fetch_test_file_urls(serials_to_fetch, after_date=self.after_date)
        self.downloader.download_all(
            serial_to_files,
            station_filter=self.station_filter,
            file_filter=self.file_filter,
            limit_per_station=self.limit_per_station,
        )


class TestImage(StrEnum):
    """Custom filter for test image files."""

    WHITE_DISTORTION = "white_distortion"
    GREEN_DISTORTION = "green_distortion"
    RED_DISTORTION = "red_distortion"
    BLUE_DISTORTION = "blue_distortion"
    BLACK_DISTORTION = "black_distortion"
    CIRCLES = "test_white_1650_circle_grid_11x11_distortion_long"

    WHITE_BLACK = "white_1650_black"
    GREEN_BLACK = "green_black"
    RED_BLACK = "red_1650_black"
    BLUE_BLACK = "blue_1650_black"

    WHITE = "white"
    BLACK = "black"
    GREEN = "green"
    BLUE = "blue"
    RED = "red"

    PANEL_ACTIVE_AREA = "panel_active_area"
    PANEL_VISUAL = "panel_visual"
    POST_DISPENSE = "post_dispense"
    Summary = "Summary File"
    ALL = ""


class Station(StrEnum):
    """Custom filter for test stations."""

    USER_FOV = "USER FOV"
    IQ_TESTER = "IQ TESTER"
    WIDE_FOV = "WIDE FOV"
    OCCELOT = "occelot"
    LCOS_AA = "LCOS AA"
    EC_TESTER = "EC"


def test_image_filter(filter_enum: TestImage) -> Callable[[str], bool]:
    """Factory function to create a file filter based on ImageFilter enum."""

    def filter_func(path: str) -> bool:
        return filter_enum.value in path and path.endswith(".png")

    return filter_func


def get_oe_serials_from_csv(csv_path: str) -> list[str]:
    """Fetch OE serial numbers from a CSV file."""
    df = pd.read_csv(csv_path)
    return df["OE serial"].dropna().astype(str).str.strip().unique().tolist()
