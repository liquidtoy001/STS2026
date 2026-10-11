"""Load the definitions of notebooks/vits-full-train.ipynb without running data loading or training.

Executes every import statement and every function/class definition of the code cells, in
order, into a namespace. Module-level settings (data paths, arms, the model instance) are not
executed; callers set any globals they need, e.g. PAD_ID.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "vits-full-train.ipynb"
KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)


def code_cells(source=None):
    """Return the code-cell sources of the notebook (or of a notebook JSON string)."""
    notebook = json.loads(source if source is not None else NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def load_definitions(namespace=None, source=None):
    namespace = {} if namespace is None else namespace
    namespace.setdefault("__name__", "sts_notebook")
    for index, cell in enumerate(code_cells(source)):
        tree = ast.parse(cell)
        nodes = [node for node in tree.body if isinstance(node, KEEP)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), f"notebook_cell_{index}", "exec"), namespace)
    return namespace


def literal_assignment(name, source=None):
    """Return the value of a plain-literal module assignment such as ARMS."""
    for cell in code_cells(source):
        for node in ast.parse(cell).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
                return ast.literal_eval(node.value)
    raise KeyError(name)
