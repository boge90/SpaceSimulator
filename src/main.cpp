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
#include "../include/MpiHelper.hpp"

int main(int argc, char **args){	
	// Reading input params
	Config *config = new Config(argc, args);

	// Initialize GPU(s)
	bool mpiSuccessful = MpiHelper::init(argc, args, config);
	bool cudaSuccessful = CudaHelper::init(config);
	
	// Exit if CUDA or MPI fails
	if(!cudaSuccessful || !mpiSuccessful){
		return EXIT_FAILURE;
	}
	
	if(config->isMaster()){	
		GLFWwindow *window = OpenGlHelper::init(config);
	
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

		// Cleaning up master memory
		delete bodies;
		delete simulator;
	}else{
		while(1){
			std::cout << "Slave " << *config->getMpiRankPtr() << std::endl;
		}
	}
	
	// Memory cleanup
	delete config;
	
	// Exit application
	return EXIT_SUCCESS;
}
