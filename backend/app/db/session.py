# This file sets up the database session for the application using SQLAlchemy. It creates an engine and a sessionmaker that can be used to interact with the database.
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Création de l'engine SQLAlchemy en utilisant l'URL de la base de données définie dans les paramètres de configuration.
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)