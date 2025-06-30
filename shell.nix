{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  packages = [
    pkgs.python312
    pkgs.uv
  ];

  shellHook = ''
    echo "Entering Python development environment."
    VENV_DIR=".venv"

    export UV_PYTHON="$(which python3)"

    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating uv virtual environment in $VENV_DIR..."
      uv venv "$VENV_DIR"
    fi

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    if [ -f "requirements.txt" ]; then
      echo "Installing dependencies from requirements.txt with uv..."
      uv pip install -r requirements.txt
    fi
  '';
}
