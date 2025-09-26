from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from rag_project.constants import AZURE_GIT_URL, AZURE_RAW_DATA_DIR
from rag_project.logger import logger


def clone_and_filter_docs(repo_url: str, target_dir: str):
    """
    Clone Azure docs repository, filter MD files and move them to target directory.

    Args:
        repo_url: URL of the Azure docs repository
        target_dir: Directory where MD files should be stored

    Returns:
        bool: True if successful, False otherwise
    """
    target_path = Path(target_dir)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Created temporary directory: %s", temp_dir)

            logger.info("Cloning repository...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--sparse",
                    repo_url,
                    temp_dir,
                ],
                check=True,
                capture_output=True,
            )

            subprocess.run(
                ["git", "sparse-checkout", "init", "--cone"],
                check=True,
                cwd=temp_dir,
                capture_output=True,
            )

            subprocess.run(
                ["git", "sparse-checkout", "set", "articles"],
                check=True,
                cwd=temp_dir,
                capture_output=True,
            )

            subprocess.run(
                ["git", "checkout"],
                check=True,
                cwd=temp_dir,
                capture_output=True,
            )

            target_path.mkdir(parents=True, exist_ok=True)

            articles_dir = Path(temp_dir) / "articles"
            if not articles_dir.exists():
                logger.error("Error: Articles directory not found in cloned repository")
                return False

            logger.info("Copying markdown files...")
            copied_files = 0
            for md_file in articles_dir.rglob("*.md"):
                try:
                    destination = target_path / md_file.name

                    counter = 1
                    while destination.exists():
                        stem = md_file.stem
                        destination = target_path / f"{stem}_{counter}{md_file.suffix}"
                        counter += 1

                    shutil.copy2(md_file, destination)
                    copied_files += 1
                    if copied_files % 100 == 0:
                        logger.info("Copied %d files.", copied_files)
                except Exception as e:
                    logger.error("Error copying %s: %s", md_file, e)

            logger.info("Successfully copied %d markdown files to: %s", copied_files, target_dir)
            return True

    except subprocess.CalledProcessError as e:
        logger.error("Git operation failed: %s", e)
        return False
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return False


def main():
    if not clone_and_filter_docs(AZURE_GIT_URL, AZURE_RAW_DATA_DIR):
        sys.exit(1)


if __name__ == "__main__":
    main()
