{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    gcc
    gnumake
    swayimg

    # Python packages from nixpkgs
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.pandas
    python312Packages.scikit-learn
    python312Packages.matplotlib
    python312Packages.seaborn
    python312Packages.pyyaml
    python312Packages.geopandas
    python312Packages.xarray
    python312Packages.rasterio
    python312Packages.pybind11
    python312Packages.torch-geometric

    # PyTorch with CUDA
    python312Packages.torch
    python312Packages.torchvision

    cudaPackages.cudatoolkit  
    cudaPackages.nccl 
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
    pkgs.zlib
    pkgs.libffi
    pkgs.cudaPackages.cudatoolkit
  ] + ":/run/opengl-driver/lib";

  shellHook = ''
    export PROJECT_ROOT="$(pwd)"
    export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
    export PATH="$CUDA_HOME/bin:$PATH"

    echo "[NIX-SHELL] PROJECT_ROOT set to: $PROJECT_ROOT" 
    
    echo "[NIX-SHELL] initializing python environment..."

    if [ ! -d ".venv" ]; then 
      echo "[NIX-SHELL] creating new virtual environment..."
      python -m venv .venv 
    fi 

    if [ -n "$VIRTUAL_ENV" ]; then 
      deactivate 
    fi

    source .venv/bin/activate
    echo "[NIX-SHELL] activated virtual environment: $VIRTUAL_ENV"
    
    echo "[NIX-SHELL] Installing remaining packages via pip"
    pip install --upgrade pip
    
    # Only install packages not available in nixpkgs
    pip install gpytorch  # if not available in nixpkgs
    
    echo "[NIX-SHELL] Installing CUDA XGBoost"
    pip install --upgrade pip
    pip install --no-cache-dir 'xgboost>=2.0.0' --config-settings=use_cuda=ON --config-settings=use_nccl=ON
    pip install -e .

    echo "[NIX-SHELL] creating project directories outside git repo"
    mkdir -p data/climate data/census data/geography 

    echo "[NIX-SHELL] Injecting pybind11 include path into .clangd"
    PYBIND11_INC=$(python -c "import pybind11; print('-I' + pybind11.get_include())") 
    echo "CompileFlags:" > .clangd  
    echo "  Add:" >> .clangd 
    echo "    - \"$PYBIND11_INC\"" >> .clangd 
    echo "    - \"-I${pkgs.python312}/include/python3.14\"" >> .clangd 
  '';
}
