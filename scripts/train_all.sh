#!/usr/bin/env bash
set -euo pipefail

uv run python -m training.train --model random_forest
uv run python -m training.train --model svm
uv run python -m training.train --model gradient_boosting
