import os
import sys

# The scripts import each other by module name, as installed side by side in lib/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
