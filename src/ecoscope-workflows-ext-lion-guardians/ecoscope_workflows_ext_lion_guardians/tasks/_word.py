# prepare custom widget list
from __future__ import annotations
from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Mapping, Hashable, TypeGuard, Union
from typing import Annotated, Any, Literal, Tuple
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema


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
Predicate = Tuple[str, Literal["=", "!=", "in", "not in"], Any]

WidgetSingle = DocHeadingWidget | DocTableWidget | DocFigureWidget
WidgetOrList = WidgetSingle | list[DocWidget]
WidgetMap = Mapping[Hashable, object]  # values may be widgets or lists/maps (we'll inspect at runtime)
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


class DocGroup(BaseModel):
    """
    First-class grouped doc widget: a group of widgets tied to filter predicates.
    `label` is optional; if omitted we'll derive something sensible from predicates.
    """
    predicates: list[Predicate]
    widgets: list[object]  
    label: str | None = None

def _is_group_tuple(x: Any) -> bool:
    # current legacy payloads look like: ((predicates...), widget_or_list)
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], (list, tuple))

def _is_widget_by_type(x: Any) -> bool:
    """
    Check if an object is a widget by examining its class name and attributes.
    This is more robust than isinstance checks when dealing with imports from different modules.
    """
    if hasattr(x, '__class__'):
        class_name = x.__class__.__name__
        if class_name in ('DocHeadingWidget', 'DocTableWidget', 'DocFigureWidget'):
            return True
        # Also check if it has the expected attributes
        if hasattr(x, 'heading') and hasattr(x, 'level'):
            if hasattr(x, 'filepath'):  # DocFigureWidget
                return True
            elif hasattr(x, 'df'):  # DocTableWidget
                return True
            else:  # DocHeadingWidget
                return True
    return False

def _coerce_widget_like(obj: object) -> object:
    """
    Turn dicts or list-of-(key,value) tuples into local widget models.
    Leaves true widget instances (local or cross-module) untouched.
    """
    # Already a widget (local or cross-module via duck typing)?
    if _is_widget_by_type(obj) or isinstance(obj, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        return obj

    # dict -> decide which widget to build
    if isinstance(obj, dict):
        d = obj  # type: dict[str, object]
        if "filepath" in d:
            return DocFigureWidget(**d)  # type: ignore[arg-type]
        if "df" in d:
            return DocTableWidget(**d)   # type: ignore[arg-type]
        if "heading" in d or "level" in d:
            return DocHeadingWidget(**d) # type: ignore[arg-type]
        return obj

    # list of (k, v) -> make a dict -> build widget
    if isinstance(obj, list) and all(isinstance(t, tuple) and len(t) == 2 for t in obj):
        d = dict(obj)  # type: dict[str, object]
        return _coerce_widget_like(d)

    return obj


def _is_widget(x: object) -> TypeGuard[WidgetSingle]:
    """
    Type guard that checks if an object is a widget, handling cross-module imports.
    """
    # First try isinstance for local widgets
    if isinstance(x, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        return True
    
    # Then use duck typing for widgets from other modules
    return _is_widget_by_type(x)


def _is_group_tuple(x: Any) -> bool:
    # current legacy payloads look like: ((predicates...), widget_or_list)
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], (list, tuple))


def _coerce_to_docgroup(x: Any) -> DocGroup | Any:
    """
    Accept legacy tuple format and convert to DocGroup, or return individual widgets as-is.
    Legacy examples:
      ((('TemporalGrouper_%B', '=', 'February'),), DocFigureWidget(...))
      ([('field','in',['A','B'])], [DocTableWidget(...), DocFigureWidget(...)])
    """
    # Handle individual widgets directly - use flexible checking
    if _is_widget_by_type(x) or isinstance(x, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        return x
    
    if not _is_group_tuple(x):
        return x
        
    preds_raw, widgets_raw = x

    # normalize predicates -> list[Predicate]
    preds: list[Predicate] = []
    if isinstance(preds_raw, tuple):
        # either a single predicate triple or a tuple of triples
        if len(preds_raw) == 3 and isinstance(preds_raw[1], str):
            preds = [preds_raw]  # single predicate
        else:
            preds = list(preds_raw)
    elif isinstance(preds_raw, list):
        preds = preds_raw
    else:
        raise TypeError(f"Unsupported predicate structure: {type(preds_raw)}")

    # normalize widgets -> list[widgets]
    if _is_widget_by_type(widgets_raw) or isinstance(widgets_raw, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        widgets = [_coerce_widget_like(widgets_raw)]  # type: list[object]
    elif isinstance(widgets_raw, list):
        widgets = []
        for widget in widgets_raw:
            w = _coerce_widget_like(widget)
            if _is_widget_by_type(w) or isinstance(w, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
                widgets.append(w)
            # else: ignore unrecognized members gracefully
    else:
        # final fallback: try to coerce once
        coerced = _coerce_widget_like(widgets_raw)
        if _is_widget_by_type(coerced) or isinstance(coerced, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
            widgets = [coerced]
        else:
            raise TypeError(f"Unsupported widget structure in group: {type(widgets_raw)}")

    # try to derive a friendly label, e.g., "February" when you group by month
    derived_label = None
    if len(preds) == 1:
        field, op, val = preds[0]
        if isinstance(val, (str, int)):
            derived_label = str(val)

    return DocGroup(predicates=preds, widgets=widgets, label=derived_label)

def _flatten_doc_items(items: Any) -> list[Any]:
    """
    Flatten nested lists/tuples but do NOT break widget objects.
    We'll handle tuple->DocGroup coercion separately.
    """
    out: list[Any] = []
    if isinstance(items, list):
        for x in items:
            out.extend(_flatten_doc_items(x))
    else:
        out.append(items)
    return out

@task
def prepare_widget_list(
    widgets: Union[
        DocWidget,                 # single widget
        list[DocWidget],           # list of widgets
        Mapping[Hashable, object], # dict-like containers of widgets/lists
        KeyValList,                # list of (key, value) pairs
        object                     # fallback to prevent Pydantic pre-validation
    ]
) -> list[DocWidget]:
    """
    Normalize widgets into a flat list[DocWidget].
    Accepts:
      - a single widget (DocHeadingWidget | DocTableWidget | DocFigureWidget)
      - a list of widgets
      - a mapping: key -> widget or list-of-widgets
      - a list of (key, value) pairs
    """

    def _flatten_values(vals: list[object]) -> list[DocWidget]:
        out: list[DocWidget] = []
        for v in vals:
            v2 = _coerce_widget_like(v)
            if _is_widget_by_type(v2) or isinstance(v2, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
                out.append(v2)  # type: ignore[arg-type]
            elif isinstance(v2, list):
                for item in v2:
                    item2 = _coerce_widget_like(item)
                    if _is_widget_by_type(item2) or isinstance(item2, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
                        out.append(item2)  # type: ignore[arg-type]
            elif isinstance(v2, Mapping):
                out.extend(_flatten_values(list(v2.values())))
        return out

    # single widget
    if _is_widget_by_type(widgets) or isinstance(widgets, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        return [widgets]  # type: ignore[list-item]

    # mapping of widgets / lists
    if isinstance(widgets, Mapping):
        return _flatten_values(list(widgets.values()))

    # list / iterable
    if isinstance(widgets, list):
        # list of (k, v) items
        if widgets and isinstance(widgets[0], tuple) and len(widgets[0]) == 2:
            return _flatten_values([dict(widgets)])
        return _flatten_values(widgets)

    # last-ditch: try to coerce once
    coerced = _coerce_widget_like(widgets)
    if _is_widget_by_type(coerced) or isinstance(coerced, (DocHeadingWidget, DocTableWidget, DocFigureWidget)):
        return [coerced]  # type: ignore[list-item]

    return []
    
def add_table(doc, table_widget, table_index):
    if table_widget.heading:
        doc.add_heading(table_widget.heading, level=table_widget.level)
    df = table_widget.df
    table = doc.add_table(rows=len(df.index) + 1, cols=len(df.columns) + 1)
    header_cells = table.rows[0].cells
    for i, column in enumerate(df.columns):
        header_cells[i + 1].text = str(column)

    for i, idx in enumerate(df.index):
        row_cells = table.rows[i + 1].cells
        # Add index
        row_cells[0].text = str(idx)

        # Add row data
        for j, value in enumerate(df.iloc[i]):
            row_cells[j + 1].text = str(value)

    doc.add_paragraph(
        f"Table {table_index}: {table_widget.caption}" if table_widget.caption else f"Table {table_index}"
    )


def add_figure(doc, widget, index):
    from docx.shared import Inches

    if widget.heading:
        doc.add_heading(widget.heading, level=widget.level)
    doc.add_picture(widget.filepath, width=Inches(widget.width))

    doc.add_paragraph(f"Figure {index}: {widget.caption}" if widget.caption else f"Figure {index}")


@task
def gather_document(
    title: Annotated[str, Field(description="The document title")],
    time_range: Annotated[TimeRange | SkipJsonSchema[None], Field(description="Time range filter")],
    # Accept anything, then coerce to supported types inside the function.
    doc_widgets: Annotated[
        list[Any], Field(description="List of document components to gather (widgets or grouped widgets)")
    ],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    filename: Annotated[
        str, Field(description="The filename to save the document as, without extension. The extension will be .docx")
    ],
    logo_path: Annotated[str | SkipJsonSchema[None], Field(description="The logo file path")] = None,
) -> Annotated[str, Field(description="The saved file path")]:
    import os
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()

    # --- Title
    heading1 = doc.add_heading(title, level=1)
    heading1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading1.paragraph_format.space_before = Inches(2)
    for run in heading1.runs:
        run.font.size = Pt(30)

    # --- Time range (optional)
    if time_range:
        fmt = time_range.time_format
        formatted_time_range = f"From {time_range.since.strftime(fmt)} to {time_range.until.strftime(fmt)}"
        heading2 = doc.add_heading(formatted_time_range, level=2)
        heading2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in heading2.runs:
            run.font.size = Pt(15)

    # --- Logo (optional)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Inches(3)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    if logo_path:
        run.add_picture(logo_path, width=Inches(1.5))

    doc.add_page_break()

    # ---- Normalize inputs:
    #  - flatten nested lists
    #  - coerce legacy group tuples into DocGroup
    normalized = [_coerce_to_docgroup(x) for x in _flatten_doc_items(doc_widgets)]

    table_index = 0
    figure_index = 0

    for item in normalized:
        # 1) First-class grouped block
        if isinstance(item, DocGroup):
            if item.label:
                doc.add_heading(str(item.label), level=2)
            for w in _flatten_doc_items(item.widgets):
                if isinstance(w, DocHeadingWidget):
                    doc.add_heading(w.heading, level=w.level)
                elif isinstance(w, DocTableWidget):
                    table_index += 1
                    add_table(doc, w, table_index)
                elif isinstance(w, DocFigureWidget):
                    figure_index += 1
                    add_figure(doc, w, figure_index)
                else:
                    # If something unexpected slips through, ignore gracefully.
                    continue

        # 2) Plain widgets (no grouping)
        elif isinstance(item, DocHeadingWidget):
            doc.add_heading(item.heading, level=item.level)
        elif isinstance(item, DocTableWidget):
            table_index += 1
            add_table(doc, item, table_index)
        elif isinstance(item, DocFigureWidget):
            figure_index += 1
            add_figure(doc, item, figure_index)

        # 3) Legacy tuple (if user passes one at the top-level and coercion returned it unmodified)
        elif _is_group_tuple(item):
            # Shouldn’t happen because we coerce above, but keep a fallback:
            coerced = _coerce_to_docgroup(item)
            if isinstance(coerced, DocGroup):
                if coerced.label:
                    doc.add_heading(str(coerced.label), level=2)
                for w in _flatten_doc_items(coerced.widgets):
                    if isinstance(w, DocHeadingWidget):
                        doc.add_heading(w.heading, level=w.level)
                    elif isinstance(w, DocTableWidget):
                        table_index += 1
                        add_table(doc, w, table_index)
                    elif isinstance(w, DocFigureWidget):
                        figure_index += 1
                        add_figure(doc, w, figure_index)

        else:
            # Unknown thing — skip
            continue

    # Handle file://
    if root_path.startswith("file://"):
        root_path = root_path[7:]

    path = os.path.join(root_path, f"{filename}.docx")
    doc.save(path)
    return path
