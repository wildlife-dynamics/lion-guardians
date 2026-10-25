import pytest
from docxtpl import DocxTemplate
from PIL import Image
from wt_registry.jsonschema import jsonschema_from_task_func

from ecoscope_workflows_ext_lion_guardians.tasks.io import LION_GUARDIANS_FILES, load_lion_guardians_file


@pytest.mark.parametrize("name", [n for n in LION_GUARDIANS_FILES if n.endswith(".docx")])
def test_report_templates_load(name):
    DocxTemplate(load_lion_guardians_file(name)).get_undeclared_template_variables()


def test_logo_loads():
    with Image.open(load_lion_guardians_file("lion-guardians.png")) as im:
        assert im.size[0] > 0


def test_unknown_file_rejected():
    with pytest.raises(ValueError, match="Unknown Lion Guardians file"):
        load_lion_guardians_file("../../secrets.txt")


def test_filename_is_an_enum_in_the_schema():
    schema = jsonschema_from_task_func(load_lion_guardians_file)
    assert schema["properties"]["filename"]["enum"] == list(LION_GUARDIANS_FILES)
