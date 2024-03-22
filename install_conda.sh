#!/bin/bash

# Function to download Miniconda installer
download_miniconda() {
    if [ ! -f "$1" ]; then
        echo "Downloading Miniconda installer..."
        wget -O "$1" "$2"
    else
        echo "Miniconda installer already exists."
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "Installing Miniconda..."
    bash "$1" -b -p "$2"
    export PATH="$2/bin:$PATH"
}

# Function for Conda initialization
conda_init_shell() {
    echo "Initializing Conda for $1..."
    eval "$(conda shell.$1 hook)"
}
translate_path() {
    if [ "$OS" = "Windows" ]; then
        echo "/$(echo $1 | sed 's|:\\|/|g' | sed 's|\\|/|g' | tr '[:upper:]' '[:lower:]')"
    else
        echo "$1"
    fi
}

# Determine OS
OS="Unknown"
case "$(uname -s)" in
    Linux*)     OS="Linux";;
    Darwin*)    OS="MacOS";;
    MINGW64*)   OS="Windows";;
    *)          echo "Unsupported OS."; exit 1;;
esac

echo "Detected OS: $OS"

# Variables based on OS
if [ "$OS" = "Windows" ]; then
    MINICONDA_INSTALLER="Miniconda3-latest-Windows-x86_64.exe"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
    MINICONDA_PATH="/c/Miniconda"  # Assuming Miniconda is installed at the root of C: drive
    WORKSPACE_PATH="/c/opt/workspace"
    SHELL_TYPE="bash"
    SCRIPTS_SUBDIR="Scripts"
else
    MINICONDA_INSTALLER="Miniconda3-latest-${OS}-x86_64.sh"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
    MINICONDA_PATH="/opt/miniconda"
    WORKSPACE_PATH="/opt/workspace"
    SHELL_TYPE="bash"
    SCRIPTS_SUBDIR="bin"
fi

# Check and create workspace directory
if [ ! -d "$WORKSPACE_PATH" ]; then
    mkdir -p "$WORKSPACE_PATH"
fi
cd "$WORKSPACE_PATH"

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    # Conda is not installed, proceed with steps 3, 4, 5
    download_miniconda "$MINICONDA_INSTALLER" "$MINICONDA_URL"
    install_miniconda "$MINICONDA_INSTALLER" "$MINICONDA_PATH"
    conda_init_shell "$SHELL_TYPE"
else
    echo "Conda is already installed."
fi

# 2. Check if Conda environment exists
ENV_NAME="mmdtc"
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    # Create Conda environment
    conda create -n "$ENV_NAME" python=3.8 -y
fi

# Finding Conda environment path
# Use the translate_path function to convert paths when necessary
ENV_PATH=$(translate_path "$(conda info --envs | grep "^$ENV_NAME\s" | awk '{print $2}')")
MINICONDA_PATH=$(translate_path "$MINICONDA_PATH")
WORKSPACE_PATH=$(translate_path "$WORKSPACE_PATH")
PYTHON_PATH="$ENV_PATH/$SCRIPTS_SUBDIR/python"
PIP_PATH="$ENV_PATH/$SCRIPTS_SUBDIR/pip"

echo $ENV_PATH
# Activate Conda environment
conda_init_shell "$SHELL_TYPE"
conda activate "$ENV_NAME"

git pull


echo "Using $PIP_PATH"
# Install packages and dependencies using the full pip path
$PIP_PATH install --upgrade pip
# Conditional installation of PaddlePaddle based on OS
if [ "$OS" = "MacOS" ]; then
    $PIP_PATH  install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
else
    $PIP_PATH  install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
fi
$PIP_PATH  install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
$PIP_PATH  install -v -e .
$PIP_PATH install future tensorboard  argparse scikit-multilearn funcy scikit-learn
$PIP_PATH install wandb
echo $PIP_PATH
echo $PYTHON_PATH
