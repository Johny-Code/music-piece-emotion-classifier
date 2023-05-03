import json


def read_json_config(*config_files):
    data = {}
    for file in config_files:
        config = open(file)
        data.update(json.load(config))
    return data
