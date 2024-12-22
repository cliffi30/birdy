import yaml


class Config:
    def __init__(self, config_path='configs/config.yml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.openai_api_key = self.config['openai']['api_key']
