import os
from daphne.cli import CommandLineInterface

# Add these lines before running the CLI
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Initialize Django (add this line)
import django
django.setup()

cli = CommandLineInterface()
cli.run(['-b', '0.0.0.0', '-p', '8000', 'backend.asgi:application'])