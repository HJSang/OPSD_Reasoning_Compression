set -x
set -e

# Determine repo root (two levels up from scripts/sft/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Use the image's pre-installed verl / torch / sglang / sgl-kernel as-is.
# The dual-path math_verify scorer lives in
# workspace/src/rewards/dual_path_math_verify.py and is wired in via verl's
# custom_reward_function.path, so no overlay-copy into the verl site-packages
# is needed.
#
# workspace/requirements.txt pins the full stack for fresh installs outside
# the image; here we only add the two extras the image doesn't ship, so we
# don't trample the image's GPU-matched torch/sglang builds.
pip install math-verify tensordict
# pip install -r requirements.txt

python -c "import verl; print(f'verl version: {verl.__version__}')"
pip freeze | grep -E "^(verl|torch|sglang|sgl-kernel|transformers|math-verify|tensordict)="
