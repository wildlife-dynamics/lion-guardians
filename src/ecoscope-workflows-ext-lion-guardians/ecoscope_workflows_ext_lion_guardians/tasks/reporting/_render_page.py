import os
from pathlib import Path
from docxtpl import DocxTemplate
from wt_registry import register
from typing import Any, Dict, Optional
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from ecoscope_workflows_ext_ste.tasks.reporting._mapbook_context import (
    DEFAULT_IMAGE_EXTENSIONS,
    build_docx_context,
    _default_filename,
)


@register()
def render_docx_page(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    filename: Optional[str] = None,
    strict_images: bool = False,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """Render one docx page from a template and context and return its path.

    Shared by the guardians, vehicles and collared lions reports, which
    previously each carried their own copy of this rendering logic.

    strict_images=False (default): missing/unreadable images are logged and
    rendered as blank slots. strict_images=True: they raise instead.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if context is None:
        raise ValueError("context cannot be None — pass an empty dict {} if no images are needed.")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = _default_filename(context)
    output_path = Path(output_dir) / filename

    # --- image validation: warn by default, raise only in strict mode ---
    for field_name, value in context.items():
        if not isinstance(value, str) or not value:
            continue
        normalized = remove_file_scheme(value)
        if Path(normalized).suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            if not os.path.exists(normalized):
                msg = f"{field_name}: image not found: {normalized}"
                if strict_images:
                    raise FileNotFoundError(msg)
                print(msg)

    try:
        tpl = DocxTemplate(template_path)
    except Exception as e:
        raise ValueError(f"Failed to load template {template_path}: {e}") from e

    rendered_context = build_docx_context(
        context=context,
        template=tpl,
        box_h_cm=box_h_cm,
        box_w_cm=box_w_cm,
    )

    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save {output_path}: {e}") from e

    print(f"Rendered page: {output_path}")
    return str(output_path)
