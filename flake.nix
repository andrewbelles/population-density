{
  description = "C++ 23 & Python 3.13 Tooling";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable"; 
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: 
    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true; 
          };
        };

        llvm   = pkgs.llvmPackages_latest; 
        python = pkgs.python313; 

      in 
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            llvm.clang 
            llvm.clang-tools 
            llvm.libcxx
            llvm.lld 
            gnumake 
            boost 
            sqlite 
            python
          ];

          shellHook = ''
            export CC=${llvm.clang}/bin/clang 
            export CXX=${llvm.clang}/bin/clang++
            export CXXFLAGS="-std=c++23 ${CXXFLAGS:-}"
            export LDFLAGGS="${LDFLAGS:-}"

            mkdir -p data/
            touch data/climate.db 
            sqlite3 data/climate.db < api_clients/climate_init.sql 

            echo [NIX] Development Toolchain Ready...
          '';
        };
      });
}
