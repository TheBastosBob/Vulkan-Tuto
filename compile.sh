#compile the .frag and .vert files into .spv files
glslangValidator   -V shaders/shader.frag -o shaders/frag.spv
glslangValidator   -V shaders/shader.vert -o shaders/vert.spv