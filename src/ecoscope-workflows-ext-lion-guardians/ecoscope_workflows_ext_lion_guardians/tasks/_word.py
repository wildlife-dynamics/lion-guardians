# prepare custom widget list 
from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Mapping, Hashable, TypeGuard, Union
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

WidgetSingle = DocHeadingWidget | DocTableWidget | DocFigureWidget
WidgetOrList = WidgetSingle | list[DocWidget]
WidgetMap = Mapping[Hashable, object]                # values may be widgets or lists/maps (we'll inspect at runtime)
KeyValList = list[tuple[Hashable, object]] 

def _is_widget(x: object) -> TypeGuard[WidgetSingle]:
    return isinstance(x, (DocHeadingWidget, DocTableWidget, DocFigureWidget))

def _flatten_values(vals: list[object]) -> list[DocWidget]:
    out: list[DocWidget] = []
    for v in vals:
        if _is_widget(v):
            out.append(v)
        elif isinstance(v, list):
            for item in v:
                if _is_widget(item):
                    out.append(item)
        elif isinstance(v, Mapping):
            out.extend(_flatten_values(list(v.values()))) 
        elif isinstance(v, tuple) and len(v) == 2 and _is_widget(v[1]):
            out.append(v[1])
        else:
            pass
    return out


@task 
def prepare_widget_list(
    widgets: Union[WidgetOrList, WidgetMap, KeyValList]
) -> list[DocWidget]:
    """
    Normalize widgets into a flat list[DocWidget].

    Accepts:
      - a single widget (DocHeadingWidget | DocTableWidget | DocFigureWidget)
      - a list of widgets
      - a mapping: key -> widget or list-of-widgets
      - a list of (key, value) pairs where value is widget or list-of-widgets
    """
    if _is_widget(widgets):
        return [widgets]
    if isinstance(widgets, Mapping):
        return _flatten_values(list(widgets.values()))  

    if isinstance(widgets, list):
        if widgets and isinstance(widgets[0], tuple) and len(widgets[0]) == 2:
            vals = [kv[1] for kv in widgets]  
            return _flatten_values(vals)
        return _flatten_values(widgets)  

    return []