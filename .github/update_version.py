"""
This script reads pyproject.toml and hijacks setuptools-scm to write the __version__.py file
on a git commit. This enables continuous development in an editable environment (i.e. no need
to keep building antidote to update the version).
"""

from pathlib import Path
from setuptools_scm import get_version, dump_version
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


root_path = Path(__file__).resolve().parent.parent
pyproject_path = root_path / "pyproject.toml"

with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)

scm_config = pyproject.get("tool", {}).get("setuptools_scm", {})

write_to = scm_config.get("write_to")

scm_version = get_version(root=root_path)

dump_version(root=root_path, version=scm_version, write_to=write_to)
