# Image-to-PlantArchitecture

## Setup

```bash
conda env create -f environment.yml -p ./env
conda activate ./env

# Build PlantString2Model
cd src/PlantString2Model
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j

# Test PlantString2Model
export DISPLAY=:10.0 # Export display for Xvfb. If you are using physical monitor, just ignore this line.
./PlantString2Model ../plantstring.txt
```

## License

This project is licensed under the MIT License.
