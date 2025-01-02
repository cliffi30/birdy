import yaml


class Config:
    def __init__(self, config_path='configs/config.yml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.openai_api_key = self.config['openai']['api_key']
            self.neo4j_uri = self.config['neo4j']['uri']
            self.neo4j_user = self.config['neo4j']['user']
            self.neo4j_password = self.config['neo4j']['password']
