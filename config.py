from typing import Any, Union, Literal

with open("config.toml", "rb") as config_file:
    import tomllib

    config: dict[str, Any] = tomllib.load(config_file)
    config_file.close()
    del config_file

debug = bool(config["general"]["debug"])
