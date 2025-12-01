{ pkgs ? import <nixpkgs> { } }:

let
	overrides = (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml));
	libPath = with pkgs;
		pkgs.lib.makeLibraryPath [
			xorg.libX11
			xorg.libXcursor
			xorg.libXi
			libxkbcommon
			xorg.libxcb
			vulkan-loader
			glfw
			libGLU
			mesa
		];
	buildInputs = with pkgs; [
		clang
		llvmPackages.bintools
		rustup
		pkg-config
		alsa-lib
		systemdLibs
		xorg.libX11
		xorg.libXcursor
		xorg.libXi
		libxkbcommon
		xorg.libxcb
		alsa-lib
		libudev-zero
		openssl
		llvm
		pkg-config
		gcc
		sqlite
		mesa
		libGLU
	];

in pkgs.mkShell rec {
	LD_LIBRARY_PATH = libPath;
	RUSTC_VERSION = overrides.toolchain.channel;
	LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
	shellHook = "
		export PATH=$PATH:\${CARGO_HOME:-~/.cargo}/bin
		export PATH=$PATH:\${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
	";
	RUSTFLAGS = (builtins.map (a: "-L ${a}/lib") [ ]);
	BINDGEN_EXTRA_CLANG_ARGS = (builtins.map (a: ''-I"${a}/include"'') [ pkgs.glibc.dev ]) ++ [
		''-I"${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${pkgs.llvmPackages_latest.libclang.version}/include"''
		''-I"${pkgs.glib.dev}/include/glib-2.0"''
		''-I"${pkgs.glib.out}/lib/glib-2.0/include/"''
	];
}
