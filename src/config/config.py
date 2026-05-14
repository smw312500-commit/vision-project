import os
from pathlib import Path

import dotenv
from flask import Config

dotenv.load_dotenv()
basedir = Path(__file__).resolve().parent.parent.parent


class BaseConfig(Config):
    REMEMBER_COOKIE_SAMESITE = "strict"
    SESSION_COOKIE_SAMESITE = "strict"


class DevelopmentConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = "sqlite:///local.sqlite"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    SECRET_KEY = os.environ.get("SECRET_KEY")
    WTF_CSRF_SECRET_KEY = os.environ.get("WTF_CSRF_SECRET_KEY")
    SECURITY_REGISTERABLE = True
    SECURITY_SEND_REGISTER_EMAIL = False
    SECURITY_PASSWORD_SALT = os.environ.get("SECURITY_PASSWORD_SALT")
    UPLOAD_FOLDER = str(Path(basedir, os.environ.get("UPLOAD_FOLDER", "uploads")))
    MODELS_FOLDER = str(Path(basedir, os.environ.get("MODELS_FOLDER", "model")))


config = {
    "development": DevelopmentConfig,
}
