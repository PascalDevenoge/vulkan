CFLAGS = -std=c++17
RELFLAGS = -Ofast
DBGFLAGS = -Og -ggdb3
LDFLAGS = -lglfw -lvulkan -lpthread -lX11 -lXxf86vm -lXrandr -lXi

debug: main.cpp
	g++ $(CFLAGS) $(DBGFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

release: main.cpp
	g++ $(CFLAGS) $(RELFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

test: debug
	./VulkanTest

clean:
	rm -f VulkanTest

.PHONY: test clean release debug
