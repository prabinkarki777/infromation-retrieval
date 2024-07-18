from flask import Flask
from config import BaseConfig


def create_app(config_class=BaseConfig):
    app = Flask(__name__)
    app.config.from_object(BaseConfig)

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    return app
