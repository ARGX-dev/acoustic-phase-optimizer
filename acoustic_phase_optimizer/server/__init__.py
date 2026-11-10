"""Local web server: JSON API plus the static front end."""

from .app import create_app

__all__ = ["create_app"]
