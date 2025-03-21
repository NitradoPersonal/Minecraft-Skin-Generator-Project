# __init__.py

# Import the main functions/classes from each file
from .app import run_web_app  # Assuming app.py has a function called start_web
from .setup import install_dependencies  # Assuming setup.py has a function called install_dependencies
from .main import train_model, generate_image  # Assuming main.py has functions called train and generate_image

# Optional: Define what gets imported when someone uses "from package import *"
__all__ = ['start_web', 'install_dependencies', 'train', 'generate_image']

