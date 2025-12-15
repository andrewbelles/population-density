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
    echo "PROJECT_ROOT set to: $PROJECT_ROOT" 
    
    echo "initializing python environment..."

    if [ ! -d ".venv" ]; then 
      echo "creating new virtual environment..."
      python -m venv .venv 
    fi 

    if [ -n "$VIRTUAL_ENV" ]; then 
      deactivate 
    fi 

    source .venv/bin/activate
    echo "activated virtual environment: $VIRTUAL_ENV"
    
    pip install --upgrade pip
    pip install numpy scipy pandas scikit-learn xgboost matplotlib seaborn 
    pip install geopandas xarray rasterio pybind11 pybind11-stubgen torch_geometric  

    echo "installing models/"
    pip install -e . 

    echo "mkdir -p data/climate data/census data/geography"
    mkdir -p data/climate data/census data/geography 
    mkdir -p models/ scripts/ analysis/ support/ 

    PYBIND11_INC=$(python -c "import pybind11; print('-I' + pybind11.get_include())") 
    echo "CompileFlags:" > .clangd  
    echo "  Add:" >> .clangd 
    echo "    - \"$PYBIND11_INC\"" >> .clangd 
    echo "    - \"-I${pkgs.python314}/include/python3.14\"" >> .clangd 
  '';
}
