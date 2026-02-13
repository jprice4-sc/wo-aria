from pathlib import Path

from download import (
    BuildMatrixParser,
    Station,
    TestFileManager,
    TestImage,
    get_oe_serials_from_csv,
    test_image_filter,
)

ROOT_DIR = Path("/Users/jprice/Documents/ADA/protoB/Artefact detection/Circles")
if not ROOT_DIR.exists():
    ROOT_DIR.mkdir(parents=True, exist_ok=True)


def download_test_files(root_dir: Path) -> None:
    """Download test files and save into standardized folders under root_dir."""

    TestFileManager(
        serial_list=None,
        source_dir=None,
        parser=None,
        output_dir=root_dir,
        station_filter=Station.USER_FOV,
        image_filter=test_image_filter(TestImage.CIRCLES),
        limit_per_station=3000,
        after_date="2025-10-22",  # Only download images after October 22, 2025
    ).run()
    print("User FOV test files downloaded successfully.")


download_test_files(ROOT_DIR)
