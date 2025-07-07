#!/bin/bash

# FUNCTIONS ---------------------------------------------------------------------
echo_help() {
	echo "Usage: $0 [options]"
	echo ""
	echo "Options:"
	echo "  -h, --help            Show this help message and exit"
	echo "  -v, --verbose         Enable verbose mode"
}

build() {
	docker compose up -d --build
}

process_deckdrafterprod() {
	python3 src/processing/process_deckdrafterprod.py
}



# CMD ARGS (main) ---------------------------------------------------------------------
set -euo pipefail

# Define the options: short -h, -v, -c [arg]; long --help, --verbose, --config [arg]
OPTIONS=hvbt::
LONGOPTS=help,verbose,build,process_deckdrafterprod::

# -temporarily store output to be able to check for errors
# -activate advanced mode with -- "$@" (preserves quoted arguments)
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
	# getopt has complained about wrong arguments to stdout
	exit 2
fi

# Use eval with set to set the parsed options into positional parameters
eval set -- "$PARSED"

# Default values
VERBOSE=false

# Parse flags
while true; do
	case "$1" in
		-h|--help)
			echo_help
			exit 0
			;;
		-v|--verbose)
			VERBOSE=true
			shift
			;;
		-b|--build)
			build
			shift
			;;
		-p|--process_deckdrafterprod)
			process_deckdrafterprod
			shift
			;;
		--)
			shift
			break
			;;
		*)
			echo "Unknown option: $1"
			exit 3
			;;
	esac
done

# Example logic
# if $VERBOSE; then
# 	echo "Verbose mode is on"
# fi
