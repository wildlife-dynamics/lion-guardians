from pathlib import Path
from docxtpl import DocxTemplate
from wt_registry import register
from typing import Any, Dict, Optional
from ecoscope_workflows_ext_custom.storage import as_readable, exists, normalize_uri, save_document
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

    Paths may be local, ``file://`` or GCS paths, so this works on desktop and on
    ecoscope-web.
    """
    if not template_path.strip():
        raise ValueError("template_path is empty")
    if not output_dir.strip():
        raise ValueError("output_dir is empty")
    if not exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if context is None:
        raise ValueError("context cannot be None — pass an empty dict {} if no images are needed.")

    if not filename:
        filename = _default_filename(context)

    # --- image validation: warn by default, raise only in strict mode ---
    for field_name, value in context.items():
        if not isinstance(value, str) or not value:
            continue
        if Path(normalize_uri(value)).suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            if not exists(value):
                msg = f"{field_name}: image not found: {value}"
                if strict_images:
                    raise FileNotFoundError(msg)
                print(msg)

    try:
        tpl = DocxTemplate(as_readable(template_path))
    except Exception as e:
        raise ValueError(f"Failed to load template {template_path}: {e}") from e

    rendered_context = build_docx_context(
        context=context,
        template=tpl,
        box_h_cm=box_h_cm,
        box_w_cm=box_w_cm,
    )

    try:
        # Escape the values, so text from event data can't alter the document's XML.
        tpl.render(rendered_context, autoescape=True)
        output_path = save_document(tpl, output_dir, filename)
    except Exception as e:
        raise ValueError(f"Failed to render or save {filename}: {e}") from e

    print(f"Rendered page: {output_path}")
    return output_path
