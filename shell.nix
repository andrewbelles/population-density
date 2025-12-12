{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python314 
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
    pip install geopandas xarray rasterio 

    echo "mkdir -p data/climate data/census data/geography"
    mkdir -p data/climate data/census data/geography 
  '';
}
