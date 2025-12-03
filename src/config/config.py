import yaml

config = {}

with open('config.yaml', 'r') as file:
   config = yaml.safe_load(file)

print(config)