# prepare custom widget list 
from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field
from ecoscope_workflows_core.annotations import AnyDataFrame

class DocHeadingWidget(BaseModel):
    heading: str | None = None
    level: int = 1


class DocTableWidget(DocHeadingWidget):
    df: AnyDataFrame
    caption: str | None = None


class DocFigureWidget(DocHeadingWidget):
    filepath: str
    caption: str | None = None
    width: float = 5.0  # Default width in inches


DocWidget = DocHeadingWidget | DocTableWidget | DocFigureWidget | list[DocTableWidget] | list[DocFigureWidget]

@task 
def prepare_widget_list(
    widgets: list[DocWidget]
) -> list[DocWidget]:
    """
    Prepare a list of document widgets, ensuring proper structure.
    """
    # Flatten and ensure proper types
    result = []
    for widget in widgets:
        if isinstance(widget, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
            result.append(widget)
        elif isinstance(widget, list):
            result.extend(widget)
    return result