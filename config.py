import os


class BaseConfig(object):
    APP_NAME = "Stock Analysis Platform"
    DEBUG = os.getenv("DEBUG", "FALSE")
