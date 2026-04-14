#!/usr/bin/env bash
set -euo pipefail

# Used in CI, extract here for readability

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "NPM Package Setup Validation"
echo "============================="
echo

check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo -e "${RED}✗ Required command not found: $1${NC}"
    exit 1
  fi
}

check_command node
check_command grep
check_command envsubst
check_command tar
check_command zip
check_command unzip

# 1. Validate wrapper script syntax
echo "1. Validating wrapper script syntax..."
if node -c npm/bin/codex-acp.js 2>/dev/null; then
  echo -e "${GREEN}✓ Wrapper script syntax is valid${NC}"
else
  echo -e "${RED}✗ Wrapper script has syntax errors${NC}"
  exit 1
fi
echo

# 2. Validate package.json files
echo "2. Validating package.json files..."
if node -e "JSON.parse(require('fs').readFileSync('npm/package.json', 'utf8'))" 2>/dev/null; then
  echo -e "${GREEN}✓ Base package.json is valid${NC}"
else
  echo -e "${RED}✗ Base package.json is invalid${NC}"
  exit 1
fi

# 3. Check template has required placeholders
echo "3. Validating template placeholders..."
missing_placeholders=0
for placeholder in PACKAGE_NAME VERSION OS ARCH; do
  if ! grep -q "\${${placeholder}}" npm/template/package.json; then
    echo -e "${RED}✗ Template missing ${placeholder} placeholder${NC}"
    missing_placeholders=1
  fi
done

if [ $missing_placeholders -eq 0 ]; then
  echo -e "${GREEN}✓ Template has all required placeholders${NC}"
else
  exit 1
fi
echo

# 4. Check version consistency
echo "4. Checking version consistency..."
CARGO_VERSION=$(grep -m1 "^version" Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
NPM_VERSION=$(node -e "console.log(require('./npm/package.json').version)")

echo "   Cargo.toml version: $CARGO_VERSION"
echo "   npm package.json version: $NPM_VERSION"

if [ "$CARGO_VERSION" != "$NPM_VERSION" ]; then
  echo -e "${RED}✗ Version mismatch${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Versions are in sync${NC}"
echo

# 5. Verify optional dependencies list
echo "5. Verifying platform packages..."
EXPECTED_PACKAGES=(
  "@proliferateai/codex-acp-darwin-arm64"
  "@proliferateai/codex-acp-darwin-x64"
  "@proliferateai/codex-acp-linux-arm64"
  "@proliferateai/codex-acp-linux-x64"
  "@proliferateai/codex-acp-win32-arm64"
  "@proliferateai/codex-acp-win32-x64"
)

missing_packages=0
for pkg in "${EXPECTED_PACKAGES[@]}"; do
  if ! grep -q "\"$pkg\":" npm/package.json; then
    echo -e "${RED}✗ Missing package: $pkg${NC}"
    missing_packages=1
  fi
done

if [ $missing_packages -eq 0 ]; then
  echo -e "${GREEN}✓ All platform packages listed in optionalDependencies${NC}"
else
  exit 1
fi

mismatched_versions=0
for pkg in "${EXPECTED_PACKAGES[@]}"; do
  pkg_version=$(node -p "require('./npm/package.json').optionalDependencies['$pkg']")
  if [ "$pkg_version" != "$NPM_VERSION" ]; then
    echo -e "${RED}✗ Optional dependency version mismatch for $pkg: expected $NPM_VERSION got $pkg_version${NC}"
    mismatched_versions=1
  fi
done

if [ $mismatched_versions -eq 0 ]; then
  echo -e "${GREEN}✓ All platform package versions match the base package version${NC}"
else
  exit 1
fi
echo

# 6. Verify platform package generation script end-to-end
echo "6. Validating platform package generation..."
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT
artifacts_dir="$tmp_root/artifacts"
output_dir="$tmp_root/output"
mkdir -p "$artifacts_dir"

make_tar_artifact() {
  local target="$1"
  local version="$2"
  local staging="$tmp_root/staging-$target"
  mkdir -p "$staging"
  printf 'test-binary' > "$staging/codex-acp"
  chmod +x "$staging/codex-acp"
  tar czf "$artifacts_dir/codex-acp-${version}-${target}.tar.gz" -C "$staging" codex-acp
}

make_zip_artifact() {
  local target="$1"
  local version="$2"
  local staging="$tmp_root/staging-$target"
  mkdir -p "$staging"
  printf 'test-binary' > "$staging/codex-acp.exe"
  (cd "$staging" && zip -q "$artifacts_dir/codex-acp-${version}-${target}.zip" codex-acp.exe)
}

artifact_version="$CARGO_VERSION"
make_tar_artifact "aarch64-apple-darwin" "$artifact_version"
make_tar_artifact "x86_64-apple-darwin" "$artifact_version"
make_tar_artifact "x86_64-unknown-linux-gnu" "$artifact_version"
make_tar_artifact "aarch64-unknown-linux-gnu" "$artifact_version"
make_zip_artifact "x86_64-pc-windows-msvc" "$artifact_version"
make_zip_artifact "aarch64-pc-windows-msvc" "$artifact_version"

bash npm/publish/create-platform-packages.sh "$artifacts_dir" "$output_dir" "$artifact_version"

expected_dirs=(
  "codex-acp-darwin-arm64"
  "codex-acp-darwin-x64"
  "codex-acp-linux-arm64"
  "codex-acp-linux-x64"
  "codex-acp-win32-arm64"
  "codex-acp-win32-x64"
)

for dir in "${expected_dirs[@]}"; do
  if [[ ! -f "$output_dir/$dir/package.json" ]]; then
    echo -e "${RED}✗ Missing generated package.json for $dir${NC}"
    exit 1
  fi

  if ! grep -q "\"name\": \"@proliferateai/$dir\"" "$output_dir/$dir/package.json"; then
    echo -e "${RED}✗ Generated package name mismatch for $dir${NC}"
    exit 1
  fi
done

echo -e "${GREEN}✓ Platform package generation works end-to-end${NC}"
echo

# 6. Verify platform package generation script end-to-end
echo "6. Validating platform package generation..."
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT
artifacts_dir="$tmp_root/artifacts"
output_dir="$tmp_root/output"
mkdir -p "$artifacts_dir"

make_tar_artifact() {
  local target="$1"
  local version="$2"
  local staging="$tmp_root/staging-$target"
  mkdir -p "$staging"
  printf 'test-binary' > "$staging/codex-acp"
  chmod +x "$staging/codex-acp"
  tar czf "$artifacts_dir/codex-acp-${version}-${target}.tar.gz" -C "$staging" codex-acp
}

make_zip_artifact() {
  local target="$1"
  local version="$2"
  local staging="$tmp_root/staging-$target"
  mkdir -p "$staging"
  printf 'test-binary' > "$staging/codex-acp.exe"
  (cd "$staging" && zip -q "$artifacts_dir/codex-acp-${version}-${target}.zip" codex-acp.exe)
}

artifact_version="0.11.1"
make_tar_artifact "aarch64-apple-darwin" "$artifact_version"
make_tar_artifact "x86_64-apple-darwin" "$artifact_version"
make_tar_artifact "x86_64-unknown-linux-gnu" "$artifact_version"
make_tar_artifact "aarch64-unknown-linux-gnu" "$artifact_version"
make_zip_artifact "x86_64-pc-windows-msvc" "$artifact_version"
make_zip_artifact "aarch64-pc-windows-msvc" "$artifact_version"

bash npm/publish/create-platform-packages.sh "$artifacts_dir" "$output_dir" "$artifact_version"

expected_dirs=(
  "codex-acp-darwin-arm64"
  "codex-acp-darwin-x64"
  "codex-acp-linux-arm64"
  "codex-acp-linux-x64"
  "codex-acp-win32-arm64"
  "codex-acp-win32-x64"
)

for dir in "${expected_dirs[@]}"; do
  if [[ ! -f "$output_dir/$dir/package.json" ]]; then
    echo -e "${RED}✗ Missing generated package.json for $dir${NC}"
    exit 1
  fi

  if ! grep -q "\"name\": \"@proliferateai/$dir\"" "$output_dir/$dir/package.json"; then
    echo -e "${RED}✗ Generated package name mismatch for $dir${NC}"
    exit 1
  fi
done

echo -e "${GREEN}✓ Platform package generation works end-to-end${NC}"
echo
