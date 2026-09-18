import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    task: str
    code: str
    error: str
    error_category: str
    iteration: int
    max_iterations: int
    success: bool
    history: Annotated[list, operator.add]