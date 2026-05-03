"""
refacer.metadata
~~~~~~~~~~~~~~~~
Strips all EXIF, XMP, and IPTC metadata from image files using exiftool.

exiftool is a system dependency (not a pip package).  If it is not found
on PATH this module logs a prominent warning and skips scrubbing rather
than crashing — consistent with the pipeline's resilient design.

Installation:
  macOS:   brew install exiftool
  Linux:   sudo apt install libimage-exiftool-perl
  Windows: https://exiftool.org  (add to system PATH)
"""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Checked once at import time so the warning surfaces early.
_EXIFTOOL_AVAILABLE: bool = shutil.which("exiftool") is not None

if not _EXIFTOOL_AVAILABLE:
    logger.warning(
        "┌─────────────────────────────────────────────────────────────────┐\n"
        "│  WARNING: exiftool not found on PATH — metadata will NOT be     │\n"
        "│  stripped from output images.  GPS coordinates, timestamps,     │\n"
        "│  and device identifiers may remain in output files.             │\n"
        "│                                                                 │\n"
        "│  Install exiftool and re-run to enable metadata scrubbing:     │\n"
        "│    macOS:   brew install exiftool                               │\n"
        "│    Linux:   sudo apt install libimage-exiftool-perl             │\n"
        "│    Windows: https://exiftool.org                                │\n"
        "└─────────────────────────────────────────────────────────────────┘"
    )


def is_available() -> bool:
    """Return True if exiftool is installed and on PATH."""
    return _EXIFTOOL_AVAILABLE


def scrub(image_path: str) -> bool:
    """
    Strip all metadata from *image_path* in-place using exiftool.

    Parameters
    ----------
    image_path : str
        Path to the image file to scrub.

    Returns
    -------
    bool
        True if scrubbing succeeded (or was skipped due to missing exiftool).
        False if exiftool was found but returned a non-zero exit code.

    Notes
    -----
    - Operates in-place; exiftool writes to a temp file and renames.
    - The original file with ``_original`` suffix left by exiftool is
      deleted automatically via the ``-overwrite_original`` flag.
    - If exiftool is not available this function logs a warning and
      returns True (skip, not failure) so the pipeline continues.
    """
    if not _EXIFTOOL_AVAILABLE:
        logger.debug("Skipping metadata scrub (exiftool not available): %s", image_path)
        return True

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-all=",                # remove all metadata
                "-overwrite_original",  # no _original backup file
                image_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(
                "exiftool failed on %s (exit %d): %s",
                image_path,
                result.returncode,
                result.stderr.strip(),
            )
            return False

        logger.debug("Metadata scrubbed: %s", image_path)
        return True

    except subprocess.TimeoutExpired:
        logger.error("exiftool timed out on %s", image_path)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error running exiftool on %s: %s", image_path, exc)
        return False