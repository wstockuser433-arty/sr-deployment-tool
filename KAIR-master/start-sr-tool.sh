#!/usr/bin/env bash
set -euo pipefail

# Resolve BASE_DIR to the directory this script lives in (follows symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
BASE_DIR="$(dirname "$SCRIPT_PATH")"
BACKEND_DIR="$BASE_DIR"
FRONTEND_DIR="$BASE_DIR/gui/frontend"

# Args: 1) port (default 8000)  2) conda env name (default torchclean)
PORT="${1:-8000}"
CONDA_ENV="${2:-torchclean}"

TMP_DIR="$(mktemp -d /tmp/sr-tool-launch.XXXXXX)"
BACKEND_LAUNCHER="$TMP_DIR/backend.sh"
FRONTEND_LAUNCHER="$TMP_DIR/frontend.sh"

# Shared conda-activation snippet: finds conda, activates the env, and
# exits cleanly (with one clear message, no stack trace) if the env
# doesn't exist or activation otherwise fails.
CONDA_ACTIVATE_SNIPPET='
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "Error: conda was not found on PATH. Cannot activate environment '"'"'%ENV%'"'"'."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk "{print \$1}" | grep -qx "%ENV%"; then
    echo "Error: conda environment '"'"'%ENV%'"'"' does not exist."
    echo "Create it or pass a different env name as the 2nd script argument."
    exit 1
fi

if ! conda activate "%ENV%" 2>/dev/null; then
    echo "Error: failed to activate conda environment '"'"'%ENV%'"'"'."
    exit 1
fi
'

BACKEND_CONDA_SNIPPET="${CONDA_ACTIVATE_SNIPPET//%ENV%/$CONDA_ENV}"

cat > "$BACKEND_LAUNCHER" <<EOF
#!/usr/bin/env bash
$BACKEND_CONDA_SNIPPET
cd "$BACKEND_DIR"
echo "Starting backend on http://127.0.0.1:$PORT (env: $CONDA_ENV) ..."
python -u -m uvicorn gui.backend.main:app --host 127.0.0.1 --port $PORT --reload
exec bash
EOF

cat > "$FRONTEND_LAUNCHER" <<EOF
#!/usr/bin/env bash
$BACKEND_CONDA_SNIPPET
cd "$FRONTEND_DIR"
echo "Starting frontend (env: $CONDA_ENV) ..."
npm run dev
exec bash
EOF

chmod +x "$BACKEND_LAUNCHER" "$FRONTEND_LAUNCHER"

# Find an available terminal emulator and launch each script in its own window
launch_terminal() {
    local title="$1"
    local script="$2"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- "$script"
    elif command -v konsole >/dev/null 2>&1; then
        konsole --new-tab -p tabtitle="$title" -e "$script"
    elif command -v xterm >/dev/null 2>&1; then
        xterm -T "$title" -e "$script" &
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        x-terminal-emulator -T "$title" -e "$script" &
    else
        echo "No supported terminal emulator found (tried gnome-terminal, konsole, xterm)."
        echo "Falling back to running '$title' in the background of this shell."
        "$script" &
    fi
}

echo "Launching backend in its own terminal (conda env: $CONDA_ENV, port: $PORT)..."
launch_terminal "SR Backend" "$BACKEND_LAUNCHER"

echo "Launching frontend in its own terminal (conda env: $CONDA_ENV)..."
launch_terminal "SR Frontend" "$FRONTEND_LAUNCHER"

echo ""
echo "Both processes have been launched in separate terminal windows."
echo "Close those windows (or Ctrl+C inside them) to stop each process."