#!/bin/bash
# Double-clickable Family Day launcher.
# Marks: macOS opens .command files in Terminal automatically when double-clicked.

# Resolve the directory this script lives in, regardless of where it's launched from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Hand off to the main launcher so there's a single source of truth
exec bash launch.sh
