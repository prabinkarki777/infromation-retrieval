from flask import Flask
from config import BaseConfig
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
import logging
import os


def create_app(config_class=BaseConfig):
    app = Flask(__name__)
    app.config.from_object(BaseConfig)

    # Register blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Scheduler setup
    def run_scrapy_spider():
        logger.info("Running Scrapy spider.")
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        project_path = os.path.join(project_root, 'search_engine')

        # Change the current working directory to the project path
        os.chdir(project_path)

        # Run the Scrapy command
        command = ['scrapy', 'crawl', 'search_engine_spider']
        subprocess.run(command)

    scheduler = BackgroundScheduler()
    # Schedule job to run every Sunday at 00:00 (midnight)
    scheduler.add_job(func=run_scrapy_spider, trigger=CronTrigger(
        day_of_week='sun', hour=0, minute=0))
    scheduler.start()
    logger.info("Scheduler started.")

    # Ensure the scheduler shuts down when the app exits
    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down.")

    return app
