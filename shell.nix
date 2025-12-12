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
    echo "initializing python environment..."

    deactivate 
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
  '';
}
