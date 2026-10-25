from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import Field
from wt_registry import register

# Files shipped in ecoscope_workflows_ext_lion_guardians/data/. They're part of the
# package release, so they're versioned and reviewed with it instead of being
# downloaded at run time.
LionGuardiansFile = Literal[
    "lion-guardians.png",
    "collared_lions_cover_page.docx",
    "collared_lion_subject_template.docx",
    "vehicles_cover_page.docx",
    "custom_vehicle_template.docx",
    "patrol_guardians_cover_page.docx",
    "custom_patrol_template.docx",
]
LION_GUARDIANS_FILES: tuple[str, ...] = get_args(LionGuardiansFile)


@register()
def load_lion_guardians_file(
    filename: Annotated[LionGuardiansFile, Field(description="The packaged file to use.")],
) -> Annotated[str, Field(description="Local path to the packaged file.")]:
    """Return the local path of a file shipped with ext-lion-guardians.

    Replaces downloading the report templates and logo at run time: the file is
    read from the installed package, so it works the same on desktop and on
    ecoscope-web, without network access.
    """
    if filename not in LION_GUARDIANS_FILES:
        raise ValueError(f"Unknown Lion Guardians file {filename!r}. Available: {', '.join(LION_GUARDIANS_FILES)}")
    resource = files("ecoscope_workflows_ext_lion_guardians") / "data" / filename
    # Installed packages are plain folders, so as_file yields the real file path.
    with as_file(resource) as path:
        local = Path(path)
    if not local.is_file():
        raise FileNotFoundError(f"Packaged file is missing: {local}")
    return str(local.resolve())
