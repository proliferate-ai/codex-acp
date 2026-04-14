#!/usr/bin/env bash
set -euo pipefail

# Used in CI, extract here for readability

# Script to update version in base package.json
# Usage: update-base-package.sh <version>

VERSION="${1:?Missing version}"

echo "Updating base package.json to version $VERSION..."

# Find the package.json relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_JSON="$SCRIPT_DIR/../package.json"

if [[ ! -f "$PACKAGE_JSON" ]]; then
  echo "❌ Error: package.json not found at $PACKAGE_JSON"
  exit 1
fi

node --input-type=module - "$PACKAGE_JSON" "$VERSION" <<'EOF'
import fs from "node:fs";

const [packageJsonPath, version] = process.argv.slice(2);
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
packageJson.version = version;

for (const dependencyName of Object.keys(packageJson.optionalDependencies ?? {})) {
  packageJson.optionalDependencies[dependencyName] = version;
}

fs.writeFileSync(packageJsonPath, `${JSON.stringify(packageJson, null, 2)}\n`);
EOF

echo "✅ Updated package.json:"
cat "$PACKAGE_JSON"
