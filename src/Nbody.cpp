#include "../include/Nbody.hpp"
#include "../include/NbodySystem.cuh"
#include <iostream>

Nbody::Nbody(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if(debugLevel & 16){
		std::cout << "Nbody.cpp\t\tInitializing\n";
	}

	// This
	this->dt = config->getDt();
	this->bodies = bodies;
	this->G = 6.7 * pow(10, -11);
	this->positions = (double*) malloc(bodies->size()*3*sizeof(double));
	
	// Creating temp positions array
	double *velocities = (double*) malloc(bodies->size()*3*sizeof(double));
	double *mass = (double*) malloc(bodies->size()*sizeof(double));
	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		
		glm::dvec3 position = body->getCenter();
		positions[i*3 + 0] = position.x;
		positions[i*3 + 1] = position.y;
		positions[i*3 + 2] = position.z;
		
		glm::dvec3 velocity = body->getVelocity();
		velocities[i*3 + 0] = velocity.x;
		velocities[i*3 + 1] = velocity.y;
		velocities[i*3 + 2] = velocity.z;
		
		mass[i] = body->getMass();
	}
	
	// Initializing CUDA system
	initializeNbodySystem(G, dt, positions, velocities, mass, bodies->size(), config);
	
	// Cleanup
	free(velocities);
	free(mass);
}

Nbody::~Nbody(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "Nbody.cpp\t\tFinalizing\n";
	}
	
	free(positions);
}

void Nbody::simulateGravity(void){
	if(debugLevel & 128){
		std::cout << "Nbodty.cpp\t\tsimulateGravity()" << std::endl;
	}
	
	// Updating bodies on GPU
	update(positions);
	
	// Updating bodies
	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		glm::dvec3 pos = glm::dvec3(positions[i*3 + 0], positions[i*3 + 1], positions[i*3 + 2]);
		
		body->setCenter(pos);
	}
}
