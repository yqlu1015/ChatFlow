from enum import Enum
from importlib.machinery import SourceFileLoader


def dict2str(dictionary: dict) -> str:
    """Convert a dict with the format
    {
        "1": "aaa",
        "2": "bbb"
    }
    to a str with the format
    ```
    1: aaa
    2: bbb
    ```

    """
    return "\n".join(
        [
            str(key.value) + ": " + str(value) if isinstance(key, Enum) else str(key) + ": " + str(value)
            for key, value in dictionary.items()
        ]
    )


def import_module_from_path(path: str, module_name: str = ""):
    """ Get module from a .py file PATH

    """
    module = SourceFileLoader(module_name, path).load_module()
    return module
