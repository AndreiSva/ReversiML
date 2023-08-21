{
  description = "A flake for building ReversiML";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system}; in
      {
        packages = rec {
          ReversiML = pkgs.python39Packages.buildPythonPackage rec {
            name = "reversiml";
            format = "pyproject";
            src = ./.;
            propagatedBuildInputs = with pkgs.python39Packages; [
              setuptools
              pygame
              tensorflow
              pyyaml
              h5py
              keras
              matplotlib
            ];
          };
          default = ReversiML;
        };

        apps = rec {
          game_app = flake-utils.lib.mkApp {
            drv = self.packages.${system}.ReversiML;
            name = "reversiml";
          };
          default = game_app;
        };
        
      }
    );
}
