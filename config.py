from typing import Any, Union, Literal

with open("config.toml", "rb") as config_file:
    from tomllib import load

    config: dict[str, Any] = load(config_file)
    config_file.close()
    del config_file

debug = bool(config["general"]["debug"])
