CFLAGS = -std=c++20 -Og -ggdb3
LDFLAGS = -lglfw -lvulkan -lpthread -lX11 -lXxf86vm -lXrandr -lXi

shaders = shaders/vert.spv shaders/frag.spv

VulkanTest: $(shaders) main.cpp
	g++ $(CFLAGS) $(DBGFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

%.spv: ./shaders/shader.vert ./shaders/shader.frag
	glslc ./shaders/shader.vert -o ./shaders/vert.spv
	glslc ./shaders/shader.frag -o ./shaders/frag.spv

test: VulkanTest
	./VulkanTest

clean:
	rm -f VulkanTest ./shaders/*.spv

.PHONY: test clean
