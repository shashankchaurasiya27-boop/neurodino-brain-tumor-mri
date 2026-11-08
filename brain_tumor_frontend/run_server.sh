#!/usr/bin/env bash
# Quick entry point forwarding to the master project runner
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$DIR/run_project.sh" "$@"
