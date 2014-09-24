#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string.h>

#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/BodyIO.hpp"
#include "../include/Simulator.hpp"
#include "../include/CudaHelper.cuh"
#include "../include/OpenGlHelper.hpp"

int main(int argc, char **args){	
	// Reading input params
	Config *config = new Config(argc, args);

	// Initialize GPU(s)
	bool cudaSuccessful = CudaHelper::init(config);
	GLFWwindow *window = OpenGlHelper::init(config);
	
	// Exit if CUDA system was un-successful or OpenGL fails
	if(!cudaSuccessful || window == NULL){
		return EXIT_FAILURE;
	}
	
	// Reading bodies from disk
	double time;
	std::vector<Body*> *bodies = new std::vector<Body*>();
	BodyIO::read(&time, bodies, config);
	
	// Init
	glfwDestroyWindow(window);
	Simulator *simulator = new Simulator(time, bodies, config);
	
	// Main loop
	std::cout << "main.cpp\t\tEntering main loop\n";
	while(simulator->isRunning()){		
		simulator->simulate();
	}
	
	// Writing bodies to disk
	BodyIO::write(simulator->getTime(), bodies, config);
	
	// Cleaning up memory
	delete bodies;
	delete simulator;
	delete config;
	
	// Exit application
	return EXIT_SUCCESS;
}
