"""Tests for ecoscope_workflows_ext_lion_guardians.tasks.reporting._render_page.

`render_docx_page` is a thin wrapper around the shared
`ecoscope_workflows_ext_ste...build_docx_context` / `_default_filename`
helpers, adding its own path normalization, argument validation, and
image-existence checks before delegating rendering to docxtpl.
"""

from __future__ import annotations

import os
from pathlib import Path

import docx
import pytest

from ecoscope_workflows_ext_lion_guardians.tasks.reporting._render_page import (
    render_docx_page,
)


class TestRenderDocxPage:
    def test_raises_on_empty_template_path(self, tmp_path):
        with pytest.raises(ValueError, match="template_path is empty"):
            render_docx_page(template_path="  ", output_dir=str(tmp_path), context={})

    def test_raises_on_empty_output_dir(self, make_docx_template):
        template = make_docx_template(["x"])
        with pytest.raises(ValueError, match="output_dir is empty"):
            render_docx_page(template_path=str(template), output_dir="  ", context={})

    def test_raises_file_not_found_for_missing_template(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Template file not found"):
            render_docx_page(
                template_path=str(tmp_path / "nope.docx"),
                output_dir=str(tmp_path / "out"),
                context={},
            )

    def test_raises_value_error_when_context_is_none(self, make_docx_template, tmp_path):
        template = make_docx_template(["x"])
        with pytest.raises(ValueError, match="context cannot be None"):
            render_docx_page(
                template_path=str(template),
                output_dir=str(tmp_path / "out"),
                context=None,
            )

    def test_happy_path_default_filename_from_grouper(self, make_docx_template, tmp_path, read_docx_text):
        template = make_docx_template(["Subject: {{ grouper_value }}"])
        output_dir = tmp_path / "out"

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"grouper_value": "Simba"},
        )
        assert result == str(output_dir / "simba.docx")
        assert os.path.exists(result)
        assert "Subject: Simba" in read_docx_text(Path(result))

    def test_custom_filename_overrides_default(self, make_docx_template, tmp_path):
        template = make_docx_template(["x"])
        output_dir = tmp_path / "out"

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"grouper_value": "Simba"},
            filename="custom_page.docx",
        )
        assert result == str(output_dir / "custom_page.docx")

    def test_missing_image_default_is_lenient_and_still_renders(self, make_docx_template, tmp_path, capsys):
        template = make_docx_template(["Map: {{ home_range_map }}"])
        output_dir = tmp_path / "out"
        missing_path = str(tmp_path / "does_not_exist.png")

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"home_range_map": missing_path},
            strict_images=False,
        )
        assert os.path.exists(result)
        captured = capsys.readouterr()
        assert "image not found" in captured.out

    def test_missing_image_strict_mode_raises(self, make_docx_template, tmp_path):
        template = make_docx_template(["Map: {{ home_range_map }}"])
        output_dir = tmp_path / "out"
        missing_path = str(tmp_path / "does_not_exist.png")

        with pytest.raises(FileNotFoundError, match="image not found"):
            render_docx_page(
                template_path=str(template),
                output_dir=str(output_dir),
                context={"home_range_map": missing_path},
                strict_images=True,
            )

    def test_non_string_and_empty_context_values_skipped_in_validation(self, make_docx_template, tmp_path):
        template = make_docx_template(["Count: {{ count }}"])
        output_dir = tmp_path / "out"

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"count": 5, "empty": "", "none_val": None, "zero": 0},
            strict_images=True,
        )
        assert os.path.exists(result)

    def test_valid_image_embedded_successfully(self, make_docx_template, make_png, tmp_path):
        template = make_docx_template(["Map: {{ home_range_map }}"])
        output_dir = tmp_path / "out"
        img = make_png(size=(400, 200))

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"home_range_map": str(img)},
            strict_images=True,
        )
        assert os.path.exists(result)

    def test_raises_value_error_on_unloadable_template(self, tmp_path):
        garbage = tmp_path / "garbage.docx"
        garbage.write_bytes(b"not actually a docx")
        with pytest.raises(ValueError, match="Failed to load template"):
            render_docx_page(
                template_path=str(garbage),
                output_dir=str(tmp_path / "out"),
                context={},
            )

    def test_render_failure_raises_value_error(self, tmp_path):
        output_dir = tmp_path / "out"
        template = tmp_path / "bad.docx"
        doc = docx.Document()
        doc.add_paragraph("{% for x in items %}{{ x }}")  # unterminated block
        doc.save(template)

        with pytest.raises(ValueError, match="Failed to render or save"):
            render_docx_page(
                template_path=str(template),
                output_dir=str(output_dir),
                context={"items": [1, 2, 3]},
            )

    def test_file_scheme_paths_are_normalized(self, make_docx_template, tmp_path):
        template = make_docx_template(["x"])
        output_dir = tmp_path / "out"

        result = render_docx_page(
            template_path="file://" + str(template),
            output_dir="file://" + str(output_dir),
            context={},
        )
        assert not result.startswith("file://")
        assert os.path.exists(result)

    def test_custom_box_dimensions_used_for_rendering(self, make_docx_template, make_png, tmp_path):
        template = make_docx_template(["Map: {{ pic }}"])
        output_dir = tmp_path / "out"
        img = make_png(size=(2000, 1000))

        result = render_docx_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"pic": str(img)},
            box_h_cm=3.0,
            box_w_cm=3.0,
        )
        assert os.path.exists(result)

    def test_output_dir_created_if_missing(self, make_docx_template, tmp_path):
        template = make_docx_template(["x"])
        output_dir = tmp_path / "nested" / "out"
        assert not output_dir.exists()

        render_docx_page(template_path=str(template), output_dir=str(output_dir), context={})
        assert output_dir.exists()
