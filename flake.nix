{
  description = "C++ 23 & Python 3.13 Tooling, SQLite Schema Generation";

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

        llvm   = pkgs.llvmPackages_17; 
        python = pkgs.python313; 

      in 
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            llvm.clang 
            llvm.libstdcxxClang 
            llvm.clang-tools 
            llvm.lld 
            gnumake 
            boost 
            sqlite 
            python
          ];

          shellHook = ''
          [NIX] Development Toolchain Ready...
          '';
        };
      });
}
