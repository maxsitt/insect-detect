#!/bin/bash

# Update the insect-detect software while backing up configuration files and handling local changes

# Source:   https://github.com/maxsitt/insect-detect
# License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
# Author:   Maximilian Sittinger (https://github.com/maxsitt)
# Docs:     https://maxsitt.github.io/insect-detect-docs/

# Immediately exit script on error, undefined variable, or pipe failure
set -euo pipefail

echo "==== Insect Detect Updater ===="
echo

# Check prerequisites
cd "$HOME/insect-detect" || { echo "ERROR: Directory $HOME/insect-detect not found."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: Git is required but not installed."; exit 1; }
command -v uv  >/dev/null 2>&1 || { echo "ERROR: uv is required but not installed."; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: Not in a git repository."; exit 1; }

# Restore stashed changes if an error occurs during update
cleanup_on_error() {
    echo "Error occurred during update."
    if git status --porcelain | grep -q "^UU\|^AA\|^DD\|^AU\|^UA" 2>/dev/null; then
        echo "Detected merge conflicts. Aborting git operations..."
        git merge --abort 2>/dev/null || git rebase --abort 2>/dev/null || true
    fi
    if git stash list | grep -q "Update script: local changes backup" 2>/dev/null; then
        echo "Restoring your stashed changes..."
        if ! git stash pop 2>/dev/null; then
            echo "WARNING: Could not restore stashed changes automatically."
        fi
    fi
    exit 1
}

trap cleanup_on_error ERR

# Fetch latest changes from origin and check for updates
echo "Fetching latest changes from GitHub..."
if ! git fetch origin; then
    echo "ERROR: Failed to fetch from GitHub. Please retry or check your internet connection."
    exit 1
fi

echo "Checking for updates..."
CHANGED_FILES=$(git diff --name-only origin/main)
if [[ -z "$CHANGED_FILES" ]]; then
    echo "No updates available. Your installation is up to date."
    exit 0
fi

# Warn if only config files have changed (likely just local custom settings)
NON_CONFIG_CHANGES=$(echo "$CHANGED_FILES" | grep -v "^configs/.*\.yaml$" || true)
CONFIG_CHANGES=$(echo "$CHANGED_FILES" | grep "^configs/.*\.yaml$" || true)

if [[ -z "$NON_CONFIG_CHANGES" ]]; then
    echo
    echo "Only configuration files have changes, which are probably your custom settings."
    echo "No actual software updates are available."
    echo
    read -rp "Do you still want to back up your config files and run the update? (y/N): " confirm_config
    if [[ "$confirm_config" =~ ^[Yy]$ ]]; then
        echo "Proceeding with the update..."
    else
        echo "Update cancelled. Your config files remain unchanged."
        exit 0
    fi
fi

CHANGES_COUNT=$(echo "$CHANGED_FILES" | grep -c .)
echo "Found $CHANGES_COUNT file(s) to update."
echo
echo "Files to be updated:"
git diff --name-status origin/main | sed 's/^/  /'

CONFIGS_WILL_BE_UPDATED=false
if [[ -n "$CONFIG_CHANGES" ]]; then
    CONFIGS_WILL_BE_UPDATED=true
    echo
    echo "Configuration files will be backed up before updating."
fi

echo
read -rp "Do you want to apply these updates? (y/N): " confirm_update
if [[ ! "$confirm_update" =~ ^[Yy]$ ]]; then
    echo
    echo "Update cancelled."
    exit 0
fi

# Back up existing config files
shopt -s nullglob
CONFIG_FILES=(configs/*.yaml)
shopt -u nullglob
BACKUP_DIR=""
if [[ ${#CONFIG_FILES[@]} -gt 0 ]]; then
    BACKUP_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    BACKUP_DIR="configs_backup/$BACKUP_TIMESTAMP"
    echo
    echo "Creating backup of config files..."
    mkdir -p "$BACKUP_DIR" || { echo "ERROR: Could not create '$BACKUP_DIR'."; exit 1; }
    if cp "${CONFIG_FILES[@]}" "$BACKUP_DIR/"; then
        echo "Backed up ${#CONFIG_FILES[@]} config file(s) to: $BACKUP_DIR"
    else
        echo "WARNING: Some config files could not be backed up."
    fi
fi

# Stash any local changes before applying updates
echo
echo "Applying updates..."
if ! git stash push --include-untracked --message "Update script: local changes backup"; then
    echo "ERROR: Failed to stash local changes."
    exit 1
fi

if ! git rebase origin/main; then
    echo "ERROR: Update failed due to conflicts during rebase."

    # Clean up any interrupted rebase state
    if [[ -d ".git/rebase-merge" || -d ".git/rebase-apply" ]]; then
        echo "Cleaning up interrupted rebase..."
        git rebase --abort 2>/dev/null
    fi

    # Try to restore stashed local changes
    if git stash list | grep -q "Update script: local changes backup" 2>/dev/null; then
        echo
        echo "Restoring your stashed changes..."
        if ! git stash pop 2>/dev/null; then
            echo
            echo "WARNING: Could not restore stashed changes automatically."
            echo "To restore them manually:"
            echo "  1. Run 'git stash show -p' to review your changes"
            echo "  2. Run 'git stash pop' to apply them"
            echo "  OR Run 'git stash drop' to discard them"
        fi
    fi

    echo
    echo "Update failed. Repository restored to previous state."
    echo "You may need to resolve conflicts manually before trying to update again."
    exit 1
fi

# Try to restore stashed local changes after successful rebase
if git stash list | grep -q "Update script: local changes backup" 2>/dev/null; then
    echo
    echo "Restoring your stashed changes..."
    if ! git stash pop 2>/dev/null; then
        echo
        echo "WARNING: Could not restore stashed changes automatically."
        CONFLICTED_FILES=$(git status --porcelain | grep "^UU\|^AA\|^DD\|^AU\|^UA" || true)
        if [[ -n "$CONFLICTED_FILES" ]]; then
            echo "Your local changes conflict with the update."
            echo "Conflicts detected in the following files:"
            echo "$CONFLICTED_FILES" | sed 's/^/  /'
            echo
            echo "To complete the update:"
            echo "  1. Edit the conflicted files to resolve conflicts"
            echo "  2. Run 'git add <file>' for each resolved file"
            echo "  3. Run 'git status' to verify all conflicts are resolved"
        else
            echo "This may be due to file permissions or other issues."
            echo "To restore them manually:"
            echo "  1. Run 'git stash show -p' to review your changes"
            echo "  2. Run 'git stash pop' to apply them"
            echo "  OR Run 'git stash drop' to discard them"
        fi
    fi
fi

# Sync Python packages if pyproject.toml or uv.lock changed
if echo "$CHANGED_FILES" | grep -qE "^(pyproject\.toml|uv\.lock)$"; then
    echo
    echo "Package dependencies have changed. Updating Python packages..."
    if uv sync; then
        echo "Python packages updated successfully."
    else
        echo "WARNING: Failed to update Python packages. Please run 'uv sync' manually."
    fi
fi

# Download new or updated models if models/models.json changed, or any archive is missing
if echo "$CHANGED_FILES" | grep -q "^models/models\.json$"; then
    echo
    echo "Model registry has changed. Updating detection models..."
else
    echo
    echo "Checking for missing detection models..."
fi
if uv run python models/download_models.py; then
    echo "Detection models are up to date."
else
    echo "WARNING: Failed to download one or more detection models."
    echo "You can retry manually by running 'uv run python models/download_models.py'."
fi

# Re-run SSL certificate generation if the script was updated
if echo "$CHANGED_FILES" | grep -q "^generate_ssl_certificates\.sh$"; then
    echo
    echo "SSL certificate script was updated. Regenerating SSL certificates..."
    if ! bash generate_ssl_certificates.sh; then
        echo "WARNING: Failed to regenerate SSL certificates."
        echo "You can regenerate them manually by running 'bash generate_ssl_certificates.sh'."
    fi
fi

# Re-install systemd service if the service file was updated
if echo "$CHANGED_FILES" | grep -q "^insect-detect-startup\.service$"; then
    echo
    echo "Startup service file was updated. Reinstalling systemd service..."
    if sudo cp insect-detect-startup.service /etc/systemd/system/ && sudo systemctl daemon-reload; then
        echo "Systemd service updated successfully."
    else
        echo "WARNING: Failed to reinstall systemd service."
        echo "You can reinstall it manually by running 'sudo cp insect-detect-startup.service /etc/systemd/system/'"
        echo "and then 'sudo systemctl daemon-reload'."
    fi
fi

echo
echo "Update complete!"
if [[ "$CONFIGS_WILL_BE_UPDATED" == true && -n "$BACKUP_DIR" ]]; then
    echo "Your previous config files are backed up in: $BACKUP_DIR"
    echo "Please review and merge any custom settings into the updated config files."
fi
