# coding/llm_sanitizer/python_sanitizer.py
from __future__ import annotations
import ast, re
from typing import List
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import TaskSchema, NormalizeResult, CodeStandardizer
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.utils import extract_code_block, normalize_whitespace, maybe_black

class PythonStandardizer(CodeStandardizer):
    language = "python"

    def normalize(self, raw: str, schema: TaskSchema) -> NormalizeResult:
        notes: List[str] = []
        code = extract_code_block(raw, prefer_langs=("python","py"))
        code = normalize_whitespace(code)
        code = re.sub(r"^```.*?\n|\n```$", "", code, flags=re.DOTALL)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            notes.append(f"parse failed: {e}; returning raw after whitespace normalize")
            return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=False)

        fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        cls_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        has_entry_top = schema.entry_point in fn_names

        if has_entry_top:
            notes.append(f"top-level function '{schema.entry_point}' found")
            cleaned = maybe_black(code)
            return NormalizeResult(files={schema.file_name: cleaned}, notes="\n".join(notes), ok=True)

        if schema.prefer_rename and len(fn_names) == 1:
            old = fn_names[0]
            notes.append(f"renaming single function '{old}' -> '{schema.entry_point}'")
            class Renamer(ast.NodeTransformer):
                def visit_FunctionDef(self, node: ast.FunctionDef):
                    if node.name == old:
                        node.name = schema.entry_point
                    return self.generic_visit(node)
            tree2 = Renamer().visit(tree)
            ast.fix_missing_locations(tree2)
            try:
                new_code = ast.unparse(tree2)  
            except Exception:
                new_code = code.replace(f"def {old}(", f"def {schema.entry_point}(")
            new_code = maybe_black(new_code)
            return NormalizeResult(files={schema.file_name: new_code}, notes="\n".join(notes), ok=True)

        for cls in cls_nodes:
            method_names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
            if schema.entry_point in method_names:
                notes.append(f"found method {cls.name}.{schema.entry_point}; adding thin adapter")
                adapter = (
                    f"\n\ndef {schema.entry_point}(*args, **kwargs):\n"
                    f"    return {cls.name}().{schema.entry_point}(*args, **kwargs)\n"
                )
                final = code + adapter
                final = maybe_black(final)
                return NormalizeResult(files={schema.file_name: final}, notes="\n".join(notes), ok=True)

        candidates = [n for n in fn_names if n in {"solve","solution","func","function","answer"}]
        if candidates:
            old = candidates[0]
            notes.append(f"renaming fallback '{old}' -> '{schema.entry_point}'")
            try:
                class Renamer(ast.NodeTransformer):
                    def visit_FunctionDef(self, node: ast.FunctionDef):
                        if node.name == old: node.name = schema.entry_point
                        return self.generic_visit(node)
                tree2 = Renamer().visit(tree); ast.fix_missing_locations(tree2)
                new_code = ast.unparse(tree2)
            except Exception:
                new_code = code.replace(f"def {old}(", f"def {schema.entry_point}(")
            new_code = maybe_black(new_code)
            return NormalizeResult(files={schema.file_name: new_code}, notes="\n".join(notes), ok=True)

        if schema.allow_wrapper:
            notes.append("no entry found; appending dynamic-dispatch adapter to call first callable")
            adapter = (
                f"\n\ndef {schema.entry_point}(*args, **kwargs):\n"
                f"    # fallback: try first callable in module\n"
                f"    import inspect\n"
                f"    for _name, _obj in globals().items():\n"
                f"        if callable(_obj) and _name not in ('{schema.entry_point}',):\n"
                f"            try:\n"
                f"                return _obj(*args, **kwargs)\n"
                f"            except TypeError:\n"
                f"                continue\n"
                f"    raise NameError('No suitable function for entry point')\n"
            )
            final = maybe_black(code + adapter)
            return NormalizeResult(files={schema.file_name: final}, notes="\n".join(notes), ok=True)

        return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=False)
