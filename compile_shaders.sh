#!/bin/bash

find ./src/shaders -not -name '*.spv' -type f -exec glslangValidator -V -o {}.spv {} \;
