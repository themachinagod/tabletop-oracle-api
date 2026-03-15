"""Allow running the seed CLI via ``python -m tabletop_oracle.cli.seed``.

This module delegates to :func:`tabletop_oracle.cli.seed.main` so that
the package can be executed as a script.
"""

import sys

from tabletop_oracle.cli.seed import main

sys.exit(main())
