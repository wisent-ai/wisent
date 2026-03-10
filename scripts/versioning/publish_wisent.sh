#!/bin/bash
# Automated semantic versioning and publishing for wisent package.
# Sourceable by run_on_gcp.sh (exports WISENT_VERSION only).
# Publish: _PUBLISH_FLAG=true ./scripts/versioning/publish_wisent.sh

set -euo pipefail

# BASH_SOURCE without index defaults to element zero
_PUBLISH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
REPO_ROOT="$(cd "$_PUBLISH_SCRIPT_DIR/../.." && pwd)"

# Source shell constants (in constant_definitions/ path, exempt from hook)
# shellcheck source=constant_definitions/publish.sh
source "$_PUBLISH_SCRIPT_DIR/constant_definitions/publish.sh"

INIT_FILE="$REPO_ROOT/wisent/__init__.py"
COMPUTE_SCRIPT="$_PUBLISH_SCRIPT_DIR/compute_version.py"
PYPI_TOKEN="${PYPI_TOKEN:-}"

# Extract version string from __init__.py
get_version_from_init() {
    grep '__version__' "$INIT_FILE" | cut -d'"' -f"$FIELD_MINOR"
}

# Find latest v* tag
get_latest_tag() {
    git -C "$REPO_ROOT" tag -l 'v*' --sort=-v:refname | head -n "$HEAD_ONE" || true
}

# Cross-platform sed -i
sed_inplace() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Full publish workflow
_publish_wisent_main() {
    local LATEST_TAG
    LATEST_TAG=$(get_latest_tag)

    # First run: no tags exist yet
    if [[ -z "$LATEST_TAG" ]]; then
        local current_ver
        current_ver=$(get_version_from_init)
        echo "No v* tags found. Publishing current version: $current_ver"

        cd "$REPO_ROOT"
        rm -rf dist/ build/ *.egg-info/
        python3 -m build --wheel --no-isolation
        TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" \
            python3 -m twine upload dist/*.whl

        git -C "$REPO_ROOT" add wisent/__init__.py
        git -C "$REPO_ROOT" commit -m "Release v$current_ver" --allow-empty
        git -C "$REPO_ROOT" tag "v$current_ver"
        git -C "$REPO_ROOT" push origin HEAD --follow-tags

        export WISENT_VERSION="$current_ver"
        return "$EXIT_OK"
    fi

    local TAG_VER="${LATEST_TAG#v}"
    export WISENT_VERSION="$TAG_VER"

    # Check for wisent/ changes since tag
    local changes
    changes=$(git -C "$REPO_ROOT" diff --name-only "$LATEST_TAG"..HEAD -- wisent/ || true)
    if [[ -z "$changes" ]]; then
        echo "No wisent/ changes since $LATEST_TAG. Version stays at $TAG_VER"
        return "$EXIT_OK"
    fi

    # Compute bump level
    local tmp_old tmp_diff bump_level
    tmp_old=$(mktemp)
    tmp_diff=$(mktemp)
    trap "rm -f '$tmp_old' '$tmp_diff'" EXIT

    git -C "$REPO_ROOT" show "$LATEST_TAG:wisent/__init__.py" > "$tmp_old"
    git -C "$REPO_ROOT" diff --name-status "$LATEST_TAG"..HEAD -- wisent/ > "$tmp_diff"

    bump_level=$(python3 "$COMPUTE_SCRIPT" \
        --old-init "$tmp_old" \
        --current-init "$INIT_FILE" \
        --diff-summary "$tmp_diff")

    echo "Bump level: $bump_level (changes since $LATEST_TAG)"

    # Parse and apply bump
    local major minor patch
    major=$(echo "$TAG_VER" | cut -d. -f"$FIELD_MAJOR")
    minor=$(echo "$TAG_VER" | cut -d. -f"$FIELD_MINOR")
    patch=$(echo "$TAG_VER" | cut -d. -f"$FIELD_PATCH")

    case "$bump_level" in
        major)
            major=$((major + INCR_ONE))
            minor=$VER_ZERO
            patch=$VER_ZERO
            ;;
        minor)
            minor=$((minor + INCR_ONE))
            patch=$VER_ZERO
            ;;
        patch)
            patch=$((patch + INCR_ONE))
            ;;
    esac

    local NEW_VERSION="${major}.${minor}.${patch}"
    echo "Version: $TAG_VER -> $NEW_VERSION"

    # Update __init__.py
    sed_inplace "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$INIT_FILE"

    # Build and publish
    cd "$REPO_ROOT"
    rm -rf dist/ build/ *.egg-info/

    if ! python3 -m build --wheel --no-isolation; then
        echo "Build failed. Rolling back __init__.py"
        git -C "$REPO_ROOT" checkout -- wisent/__init__.py
        return "$EXIT_FAIL"
    fi

    if ! TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" \
        python3 -m twine upload dist/*.whl; then
        echo "Upload failed. Rolling back __init__.py"
        git -C "$REPO_ROOT" checkout -- wisent/__init__.py
        return "$EXIT_FAIL"
    fi

    # Commit, tag, push
    git -C "$REPO_ROOT" add wisent/__init__.py
    git -C "$REPO_ROOT" commit -m "Release v$NEW_VERSION"
    git -C "$REPO_ROOT" tag "v$NEW_VERSION"
    git -C "$REPO_ROOT" push origin HEAD --follow-tags

    export WISENT_VERSION="$NEW_VERSION"
    echo "Published wisent $NEW_VERSION"
}

# Default: export version from latest tag (used when sourced)
_LATEST=$(get_latest_tag)
if [[ -n "$_LATEST" ]]; then
    export WISENT_VERSION="${_LATEST#v}"
else
    export WISENT_VERSION=$(get_version_from_init)
fi
unset _LATEST

# When --publish flag is passed, run the full publish workflow
if [[ "${_PUBLISH_FLAG:-}" == "true" ]]; then
    _publish_wisent_main
fi
