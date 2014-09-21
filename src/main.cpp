#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string.h>

#include "../include/Body.hpp"
#include "../include/BodyIO.hpp"
#include "../include/Simulator.hpp"
#include "../include/CudaHelper.cuh"
#include "../include/OpenGlHelper.hpp"

int main(int argc, char **args){	
	// Initialize GPU(s)
	bool cudaSuccessful = CudaHelper::init(0);
	GLFWwindow *window = OpenGlHelper::init();
	
	// Exit if CUDA system was un-successful or OpenGL fails
	if(!cudaSuccessful || window == NULL){
		return EXIT_FAILURE;
	}
	
	// Simulation parameters
	double dt = 10.0;
	for(int i=0; i<argc; i++){
		if(strcmp(args[i], "--dt") == 0){
			dt = strtod(args[++i], NULL);
		}
	}

	// Reading bodies from disk
	double time;
	std::vector<Body*> *bodies = new std::vector<Body*>();
	BodyIO::read(&time, bodies);
	
	// Init
	glfwDestroyWindow(window);
	Simulator *simulator = new Simulator(time, dt, bodies);
	
	// Main loop
	std::cout << "main.cpp\t\tEntering main loop\n";
	while(simulator->isRunning()){		
		simulator->simulate();
	}
	
	// Writing bodies to disk
	BodyIO::write(simulator->getTime(), bodies);
	
	// Cleaning up memory
	delete bodies;
	delete simulator;
	
	// Exit application
	return EXIT_SUCCESS;
}
