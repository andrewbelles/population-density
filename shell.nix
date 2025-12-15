{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python314 
    gcc 
    gnumake 
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib 
    pkgs.glibc 
    pkgs.zlib 
    pkgs.libffi 
  ];

  shellHook = ''
    export PROJECT_ROOT="$(pwd)"
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
    
    echo "[NIX-SHELL] Installing packages"
    pip install --upgrade pip
    pip install numpy scipy pandas scikit-learn xgboost matplotlib seaborn gpytorch 
    pip install geopandas xarray rasterio pybind11 pybind11-stubgen torch_geometric  

    echo "[NIX-SHELL] installing models/"
    pip install -e . 

    echo "[NIX-SHELL] creating project directories outside git repo"
    mkdir -p data/climate data/census data/geography 

    echo "[NIX-SHELL] Injecting pybind11 include path into .clangd"
    PYBIND11_INC=$(python -c "import pybind11; print('-I' + pybind11.get_include())") 
    echo "CompileFlags:" > .clangd  
    echo "  Add:" >> .clangd 
    echo "    - \"$PYBIND11_INC\"" >> .clangd 
    echo "    - \"-I${pkgs.python314}/include/python3.14\"" >> .clangd 
  '';
}
