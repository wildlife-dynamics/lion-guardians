from typing import Union
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import ViewState, LayerDefinition

@task
def view_data(val: Union[ViewState, LayerDefinition]) -> str:
    """
    Prints and returns a string representation of a ViewState or LayerDefinition object.

    Args:
        val (Union[ViewState, LayerDefinition]): The object to display.

    Returns:
        str: The string representation of the object.
    """
    val_str = str(val)
    print(f"viewing data types {val_str}")
    return val_str
