#!/bin/bash

echo 'eval "$(starship init bash)"' >> ~/.bashrc
mkdir -p ~/.config
cp .devcontainer/starship.toml ~/.config

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc