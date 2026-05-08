#!/usr/bin/env bash
set -euo pipefail

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js >=18 is required for @krxgu/kernel-skills"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required for @krxgu/kernel-skills"
  exit 1
fi

npm install
npx kernel-skills list >/dev/null
echo "kernel-skills is installed and working"
