{
  description        = "topographic dev shell"; 
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; 

  outputs            = { self, nixpkgs }: 
  let 
    system = "x86_64-linux"; 
    pkgs   = import nixpkgs {
      inherit system; 
      config.allowUnfree = true; 
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        # Interpreter + toolchain 
        python312 
        gcc 
        gnumake 
        gdal 
        proj 
        geos 

        # Python packages from nixpkgs 
        python312Packages.numpy 
        python312Packages.scipy  
        python312Packages.pandas 
        python312Packages.scikit-learn
        python312Packages.matplotlib
        python312Packages.seaborn
        python312Packages.pyyaml
        python312Packages.xarray
        python312Packages.rasterio
        python312Packages.pybind11
        python312Packages.fiona
        python312Packages.libpysal
        python312Packages.optuna
        python312Packages.umap-learn

        # Cuda 
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
        export CUDA_HOME="${pkgs.cudaPackagees.cudatoolkit}"
        export PATH="$CUDA_HOM/bin:$PATH"

        echo "[NIX-SHELL] PROJECT_ROOT set to: $PROJECT_ROOT"
        echo "[NIX-SHELL] initializing python environment..."

        if [ ! -d ".venv" ]; then 
          echo "[NIX-SHELL] creating new virtual environment..."
          python -m venv .venv 
        fi 

        if [ -n "$VIRTUAL_ENV" ]; then 
          deactivate 
        fi 

        export PATH="$PROJECT_ROOT/scripts:$PATH"

        source .venv/bin/activate 
        echo "[NIX-SHELL] activated virtual environment: $VIRTUAL_ENV"

        echo "[NIX-SHELL] installing remaining packages via pip"

        python -m pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu124 

        python -m pip install torch-geometric rasterstats pyrosm networkit imageio 

        echo "[NIX-SHELL] installing CUDA xgboost"

        pip install --upgrade pip 
        pip install --no-cache-dir xgboost \
          --config-settings=use_cuda=ON \
          --config-settings=use_nccl=ON 

        pip install -e .

        echo "[NIX-SHELL] creating project directories outside git repo"
        mkdir -p data/datasets data/tensors

        echo "[NIX-SHELL] injecting python development headers"
        PYTHON_INC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")

        export C_INCLUDE_PATH="$PYTHON_INC:''${C_INCLUDE_PATH:-}"
        export CPLUS_INCLUDE_PATH="$PYTHON_INC:''${CPLUS_INCLUDE_PATH:-}"
        export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"

        echo "[NIX-SHELL] injecting pybind11 include path into .clangd"
        PYBIND11_INC=$(python -c "import pybind11; print(pybind11.get_include())")
        TORCH_INCS=$(python -c "from torch.utils.cpp_extension import include_paths; \
          print('\n'.join(include_paths()))" 2>/dev/null || true)

        cat > .clangd << EOF
        CompileFlags:
          Add:
            - "-std=c++17"
            - "-DTORCH_EXTENSION_NAME=topo_kernels"
            - "-I${pkgs.cudaPackages.cudatoolkit}/include"
            - "-I${PYBIND11_INC}"
            - "-I${PYTHON_INC}"
        EOF 

        if [ -n "$TORCH_INC" ]; then 
          while IFS= read -r inc; do 
            echo "    - \"-I$inc\"" >> .clangd 
          done <<< "$TORCH_INCS"
        fi 
      '';
    };
  };
}
