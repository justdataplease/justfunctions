import ast
import os


def extract_docstrings(file_path):
    with open(file_path, 'r') as file:
        node = ast.parse(file.read(), filename=file_path)

    functions = {}
    classes = {}

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            functions[item.name] = ast.get_docstring(item)
        elif isinstance(item, ast.ClassDef):
            class_docstrings = {'__doc__': ast.get_docstring(item)}
            for class_item in item.body:
                if isinstance(class_item, ast.FunctionDef):
                    class_docstrings[class_item.name] = ast.get_docstring(class_item)
            classes[item.name] = class_docstrings

    return {'functions': functions, 'classes': classes}


def generate_markdown_documentation(directory_path, output_file):
    toc = []  # Table of Contents
    documentation = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.py') and filename != '__init__.py':
            file_path = os.path.join(directory_path, filename)
            docstrings = extract_docstrings(file_path)
            toc.append(f"- [{filename}](#{filename.lower().replace('.py', '').replace(' ', '-')})")
            documentation.append(f"# {filename}\n\n")
            for func_name, docstring in docstrings['functions'].items():
                toc.append(f"  - [Function: {func_name}](#{func_name.lower().replace('_', '-').replace(' ', '-')})")
                documentation.append(f"## Function: {func_name}\n")
                documentation.append(f"```\n{docstring}\n```\n\n")
            for class_name, methods in docstrings['classes'].items():
                toc.append(f"  - [Class: {class_name}](#{class_name.lower().replace('_', '-').replace(' ', '-')})")
                documentation.append(f"## Class: {class_name}\n")
                class_doc = methods.pop('__doc__', None)
                if class_doc:
                    documentation.append(f"```\n{class_doc}\n```\n\n")
                for method_name, method_doc in methods.items():
                    toc.append(f"    - [Method: {method_name}](#{method_name.lower().replace('_', '-').replace(' ', '-')})")
                    documentation.append(f"### Method: {method_name}\n")
                    documentation.append(f"```\n{method_doc}\n```\n\n")

    with open(output_file, 'w') as md_file:
        # Write Table of Contents
        md_file.write("## Table of Contents\n")
        for line in toc:
            md_file.write(line + '\n')
        md_file.write('\n---\n\n')
        # Write detailed documentation
        for line in documentation:
            md_file.write(line)


if __name__ == "__main__":
    directory_path = 'justfunctions'
    output_md_file = 'functions_documentation.md'  # The output markdown file name
    generate_markdown_documentation(directory_path, output_md_file)
