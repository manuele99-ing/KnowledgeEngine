from __future__ import annotations

from pathlib import Path

from datasets import DownloadMode, load_dataset


class HuggingFaceDownloader:
    """
    Downloads datasets from Hugging Face.

    This class is intentionally generic and can download
    any dataset supported by the Hugging Face Hub.

    Downloaded datasets are cached locally by the
    Hugging Face datasets library.
    """

    def download(
        self,
        dataset_name: str,
        dataset_root: Path,
        subset: str | None = None,
        split: str = "train",
        force: bool = False,
    ):
        """
        Downloads a dataset.

        Parameters
        ----------
        dataset_name
            Hugging Face dataset identifier.

        dataset_root
            Local cache directory.

        subset
            Optional dataset configuration.

        split
            Dataset split.

        force
            Forces a fresh download.

        Returns
        -------
        datasets.Dataset
        """

        download_mode = (
            DownloadMode.FORCE_REDOWNLOAD
            if force
            else DownloadMode.REUSE_DATASET_IF_EXISTS
        )

        return load_dataset(
            path=dataset_name,
            name=subset,
            split=split,
            cache_dir=str(dataset_root),
            download_mode=download_mode,
        )