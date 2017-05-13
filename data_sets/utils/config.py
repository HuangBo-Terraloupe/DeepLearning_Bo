import os.path
import yaml

config_dir = os.path.join(os.path.dirname(__file__), '../../data/config')
config_file = os.path.join(config_dir, 'config.yml')


def get_datadir(directories):
    ''' This function checks the given list of possible data directories and
    returns the first existing one.'''

    for directory in directories:
        if os.path.exists(directory):
            return directory


def load_config(config_file=config_file):
    return Config.load(config_file)


class Config:
    def __init__(self, attributes):
        for k, v in attributes.items():
            setattr(self, k, v)

    @staticmethod
    def load(filename):
        file = open(filename, 'r')
        config_data = yaml.load(file)
        return Config(config_data)
