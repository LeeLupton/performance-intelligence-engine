# Minimal serving image for the Rust bridge — the production-relevant artifact:
# scores exported ONNX bundles with no Python and no ONNX Runtime library.
# Multi-stage so the final image is just Debian slim + the static-ish binary.
#
# Build context is this crate directory:
#   docker build -t idr-intelligence-rt:local rust/idr-intelligence-rt
#
# The toolchain is pinned by rust-toolchain.toml (1.96.1); the base tracks the
# same minor so rustup does not need to fetch a different toolchain at build.
FROM rust:1.96-slim-bookworm AS build
WORKDIR /crate
COPY . .
RUN cargo build --release --locked --bin idr-intelligence-rt

FROM debian:bookworm-slim
RUN useradd --create-home --uid 10001 idr
COPY --from=build /crate/target/release/idr-intelligence-rt /usr/local/bin/idr-intelligence-rt
USER idr
ENTRYPOINT ["idr-intelligence-rt"]
CMD ["--help"]
