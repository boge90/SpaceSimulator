#include "../include/BodyRotator.hpp"
#include "../include/BodyRotatorSystem.cuh"
#include <iostream>

BodyRotator::BodyRotator(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->config = config;
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "BodyRotator.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->dt = config->getDt();
	this->bodies = bodies;
}

BodyRotator::~BodyRotator(){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "BodyRotator.cpp\t\tFinalizing" << std::endl;
	}
}

void BodyRotator::simulateRotation(void){
	double theta;
	glm::dvec3 center;

	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		
		theta = body->getRotation();
		theta += body->getRotationSpeed()*dt;
		
		if(theta > 2.0*M_PI){
			theta -= 2.0*M_PI;
		}
		
		body->setRotation(theta);
	}
}
