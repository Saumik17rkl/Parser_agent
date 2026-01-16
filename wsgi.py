"""
WSGI config for Math Parser Agent.

This module contains the WSGI callable as a module-level variable named `application`.
"""

import os
from bot import app as application

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    application.run(host='0.0.0.0', port=port)
