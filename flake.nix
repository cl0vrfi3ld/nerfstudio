{
  description = "ROCm development environment for nerfstudio";

  # ---------------------------------------------------------------------------
  # This flake exposes a `devshell` (numtide/devshell) that gives a functional
  # nerfstudio development environment built around AMD ROCm instead of CUDA.
  #
  # nerfstudio is upstream a CUDA/NVIDIA-only project (torch is pip-pinned to
  # `+cu118` wheels; gsplat/nerfacc/tiny-cuda-nn are CUDA kernels). This shell
  # substitutes the ROCm PyTorch stack from nixpkgs and never pulls a CUDA dep.
  #
  # Strategy (hybrid, because ~40 of nerfstudio's Python deps are not packaged
  # in nixpkgs): Nix provides the ROCm torch stack + native/system libraries +
  # the ROCm toolchain, and a startup hook bootstraps a `--system-site-packages`
  # virtualenv into which `pip install -e .` pulls the pure-Python remainder.
  # Because Nix already satisfies torch/torchvision, pip skips them -> no CUDA.
  #
  # CAVEATS (ROCm):
  #   * tiny-cuda-nn is CUDA-only with no ROCm port. Its import is already
  #     guarded in nerfstudio/utils/external.py, so tcnn-backed encodings are
  #     simply unavailable; nerfacto & friends run with the pure-torch encoders.
  #   * gsplat==1.4.0 / nerfacc==0.5.2 compile HIP kernels against the Nix torch
  #     at `pip install` time. ROCm builds are experimental and may fail on some
  #     cards; the startup hook tolerates that failure instead of breaking the
  #     shell, and prints a warning so you know which methods are unavailable.
  #   * numpy is pinned <2 inside the venv (matching upstream's Dockerfile).
  # ---------------------------------------------------------------------------

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      devshell,
    }:
    # ROCm is Linux/x86_64 only.
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        # ---- EDIT for your GPU ------------------------------------------------
        # Defaults target RDNA3 (RX 7000 series). For other cards change both,
        # e.g. RDNA2 (RX 6000): gfxArch = "gfx1030"; gfxOverride = "10.3.0";
        gfxArch = "gfx1100"; # PYTORCH_ROCM_ARCH / HCC_AMDGPU_TARGET
        gfxOverride = "11.0.0"; # HSA_OVERRIDE_GFX_VERSION
        # -----------------------------------------------------------------------

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ devshell.overlays.default ];
          config = {
            allowUnfree = true; # some ROCm components are marked unfree
            rocmSupport = true; # => torch/torchvision build against ROCm
            cudaSupport = false; # => explicitly no CUDA anywhere
          };
        };

        inherit (pkgs) lib;

        # nixos-unstable no longer ships python310 (EOL) and its python311
        # package set is partially broken (sphinx dropped 3.11). Python 3.12 is
        # the best-supported interpreter here and is fine for nerfstudio.
        python = pkgs.python312;

        # Interpreter carrying the ROCm torch stack at the Nix level; the venv
        # inherits these via --system-site-packages so pip never fetches torch.
        pythonEnv = python.withPackages (
          ps: with ps; [
            pip
            virtualenv
            setuptools
            wheel
            numpy
            torch # ROCm build (rocmSupport = true)
            torchvision # matches the ROCm torch above
          ]
        );

        # Unified ROCm toolkit tree so ROCM_PATH / HIP_PATH point at one prefix,
        # which is what the HIP build system expects when compiling extensions.
        rocmToolkit = pkgs.symlinkJoin {
          name = "rocm-toolkit";
          paths = with pkgs.rocmPackages; [
            clr
            hipcc
            rocm-runtime
            rocm-device-libs
            rocblas
            hipblas
            hipblas-common
            rocsparse
            hipsparse
            rocsolver
            hipsolver
            rocfft
            hipfft
            rocrand
            hiprand
            miopen
            rccl
            rocthrust
            rocprim
            hipcub
            rocm-smi
            rocminfo
          ];
        };

        # Native libs that pip wheels (open3d, pymeshlab, opencv, ...) dlopen at
        # runtime, plus what HIP extension builds link against.
        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glib
          libGL
          libGLU
          libx11
          libxext
          libxrender
          libsm
          libice
          libxcb
          libxau
          libxdmcp
          libxkbcommon
          fontconfig
          freetype
          dbus
          udev
          glew
          ceres-solver
          glog
          gflags
        ];
      in
      {
        devShells.default = pkgs.devshell.mkShell {
          name = "nerfstudio-rocm";

          packages =
            (with pkgs; [
              pythonEnv
              rocmToolkit
              # build toolchain for HIP extensions (gsplat/nerfacc)
              cmake
              # ninja
              gnumake
              pkg-config
              gcc
              git
              # data-processing system deps used by ns-process-data
              colmap
              ffmpeg
            ])
            ++ runtimeLibs;

          env = [
            {
              name = "ROCM_PATH";
              value = "${rocmToolkit}";
            }
            {
              name = "HIP_PATH";
              value = "${rocmToolkit}";
            }
            {
              name = "HSA_OVERRIDE_GFX_VERSION";
              value = gfxOverride;
            }
            {
              name = "PYTORCH_ROCM_ARCH";
              value = gfxArch;
            }
            {
              name = "HCC_AMDGPU_TARGET";
              value = gfxArch;
            }
            # Let pip-built extensions + wheels find ROCm/system libraries.
            {
              name = "LD_LIBRARY_PATH";
              prefix = "${rocmToolkit}/lib:${lib.makeLibraryPath runtimeLibs}";
            }
          ];

          commands = [
            {
              name = "nerfstudio-install";
              help = "(re)create .venv and pip install -e . (torch stays from Nix)";
              command = "setup_venv --force";
            }
            {
              name = "rocm-check";
              help = "print detected GPU + verify torch sees the ROCm device";
              command = ''
                ${rocmToolkit}/bin/rocminfo | grep -i -m1 gfx || true
                python -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| gpu?', torch.cuda.is_available())"
              '';
            }
          ];

          # Startup hook: bootstrap + activate the venv, install once (guarded by
          # a stamp file so a normal shell entry is fast).
          devshell.startup.venv.text = ''
            setup_venv() {
              if [ "$1" = "--force" ] || [ ! -d .venv ]; then
                ${pythonEnv}/bin/virtualenv --system-site-packages .venv
              fi
              # shellcheck disable=SC1091
              source .venv/bin/activate
              python -m pip install --upgrade pip
              python -m pip install "numpy<2"
              # torch/torchvision are already satisfied by Nix -> pip skips them,
              # so no CUDA wheels are ever downloaded.
              if python -m pip install -e .; then
                touch .venv/.nerfstudio-installed
              else
                echo "[warn] 'pip install -e .' failed (likely the gsplat/nerfacc HIP"
                echo "[warn] extension build). The rest of nerfstudio is still usable;"
                echo "[warn] see the CAVEATS comment at the top of flake.nix."
              fi
            }

            if [ ! -f .venv/.nerfstudio-installed ]; then
              setup_venv
            else
              # shellcheck disable=SC1091
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
