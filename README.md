to run locally
put opus.dll under .venv/Scripts

for pipeline

1) make sure cmake is installed

2) generate opus.dll
git clone https://gitlab.xiph.org/xiph/opus.git
cd opus
cmake -S . -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release

3) add opus.dll path to 'Path' environment variable
