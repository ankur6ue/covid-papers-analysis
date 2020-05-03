"""
Defines main entry point for ETL process.
"""

import sys
sys.path.append('/home/ankur/dev/apps/ML/cord19q/src/python/cord19q/etl')
from .execute import Execute

if __name__ == "__main__":
    if len(sys.argv) > 1:
        Execute.run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
