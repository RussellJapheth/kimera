# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Entrypoint for python -m app
"""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
