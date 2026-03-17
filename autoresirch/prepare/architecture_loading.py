from __future__ import annotations
from pathlib import Path
import ast
import importlib.util
from types import ModuleType
from .schemas import ArchitectureSpec, ArchitectureConstraints, ARCHITECTURE_CONSTRAINTS, EDITABLE_TRAIN_FILE, LoadedArchitecture, ALLOWED_TRAIN_IMPORTS, FORBIDDEN_CALL_NAMES, FORBIDDEN_ATTRIBUTE_NAMES


def validate_architecture_spec(
    spec: ArchitectureSpec,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
) -> None:
    if spec.family not in constraints.allowed_families:
        raise ValueError(f"Unsupported architecture family: {spec.family!r}")
    if spec.activation not in constraints.allowed_activations:
        raise ValueError(f"Unsupported activation: {spec.activation!r}")
    if spec.normalization not in constraints.allowed_normalizations:
        raise ValueError(f"Unsupported normalization: {spec.normalization!r}")
    if not constraints.allow_bias and spec.use_bias:
        raise ValueError("Bias parameters are disabled by ArchitectureConstraints.")
    if len(spec.hidden_dims) > constraints.max_hidden_layers:
        raise ValueError("Too many hidden layers in ArchitectureSpec.hidden_dims.")
    if any(width <= 0 or width > constraints.max_hidden_width for width in spec.hidden_dims):
        raise ValueError("ArchitectureSpec.hidden_dims contains an invalid layer width.")
    if spec.dropout < 0 or spec.dropout > constraints.max_dropout:
        raise ValueError(
            f"Dropout must be between 0 and {constraints.max_dropout}, got {spec.dropout}."
        )


def validate_train_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    has_architecture = False
    has_build_model = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_TRAIN_IMPORTS:
                    raise ValueError(f"Import not allowed in train.py: {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name not in ALLOWED_TRAIN_IMPORTS:
                raise ValueError(f"Import not allowed in train.py: {module_name!r}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ARCHITECTURE":
                    has_architecture = True
        elif isinstance(node, ast.FunctionDef) and node.name == "build_model":
            has_build_model = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"Forbidden call in train.py: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTE_NAMES:
                raise ValueError(f"Forbidden method call in train.py: .{node.func.attr}()")

    if not has_architecture:
        raise ValueError("train.py must define ARCHITECTURE.")
    if not has_build_model:
        raise ValueError("train.py must define build_model(context).")


def load_train_definition(path: Path = EDITABLE_TRAIN_FILE) -> LoadedArchitecture:
    validate_train_source(path)
    spec = importlib.util.spec_from_file_location("architecture_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _extract_loaded_architecture(module)


def _extract_loaded_architecture(module: ModuleType) -> LoadedArchitecture:
    architecture = getattr(module, "ARCHITECTURE", None)
    build_model = getattr(module, "build_model", None)

    if not isinstance(architecture, ArchitectureSpec):
        raise TypeError("train.py must define ARCHITECTURE as an ArchitectureSpec instance.")
    if not callable(build_model):
        raise TypeError("train.py must define a callable build_model(context).")

    validate_architecture_spec(architecture)
    return LoadedArchitecture(
        spec=architecture,
        build_model=build_model,
        module_name=module.__name__,
    )