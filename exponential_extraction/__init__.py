"""Allow imports from the repository root without installing the package."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

src_package_dir = Path(__file__).resolve().parent.parent / "src" / __name__
if src_package_dir.exists():
    __path__.append(str(src_package_dir))
