#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Al-Borno-Lab/MouseArmImitationLearning.git"
REPOSITORY_DIR="/opt/MouseArmImitationLearning"
CONDA_DIR="/opt/miniconda"
CONDA_ENV="MouseArmImitationLearningEnv"
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
ASSIGNMENT_PATH="/tmp/assignment.json"

sudo apt update
sudo apt install -y git wget curl

wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh

sudo bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
rm /tmp/miniconda.sh

sudo git clone "$REPOSITORY_URL" "$REPOSITORY_DIR"

sudo chown -R "$(id -u):$(id -g)" "$CONDA_DIR" "$REPOSITORY_DIR"

source "$CONDA_DIR/etc/profile.d/conda.sh"

conda config --system --remove-key channels || true
conda config --system --add channels conda-forge
conda config --system --set channel_priority strict

cd "$REPOSITORY_DIR"

conda env create -f environment.yml
conda activate "$CONDA_ENV"

pip install -U huggingface_hub

hf download AlBornoLab/MouseArmModel \
    --repo-type dataset \
    --local-dir ./models

ASSIGNMENT_BUCKET=$(curl -fsSL \
    -H "Metadata-Flavor: Google" \
    "$METADATA_URL/assignment-bucket")

ASSIGNMENT_INDEX=$(curl -fsSL \
    -H "Metadata-Flavor: Google" \
    "$METADATA_URL/assignment-index")

gcloud storage cp \
    "gs://${ASSIGNMENT_BUCKET}/assignments/${ASSIGNMENT_INDEX}.json" \
    "$ASSIGNMENT_PATH"

python cloud/cloud_worker.py "$ASSIGNMENT_PATH"

sudo shutdown -h now
