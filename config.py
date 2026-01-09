import tomllib
from typing import Any, Union, Literal

config_file = open("config.toml", "rb")
config = tomllib.load(config_file)
config_file.close()
del config_file
