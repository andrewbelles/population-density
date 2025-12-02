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
        llvmPackages = pkgs.llvmPackages;

        python = pkgs.python314; 
      in 
        {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              clang 
              clang-tools 
              gnumake 
              bear 
              boost.dev 
              sqlite.dev
              llvmPackages.libcxx
              python
            ];

            CXXFLAGS = "-std=c++23";
            NIX_CFLAGS_COMPILE = "-isystem ${llvmPackages.libcxx.dev}/include/c++/v1";
            LDFLAGS = "";

            shellHook = ''
            mkdir -p data/
            touch data/climate.db 
            sqlite3 data/climate.db < clients/climate_init.sql 

            if [ -x /run/current-system/sw/bin/clangd ]; then 
              export PATH="/run/current-system/sw/bin:$PATH"
            fi 

            echo [NIX] Development Toolchain Ready...
          '';
        };
      });
}
