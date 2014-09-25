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
	
	// Initializing CUDA System
	initializeBodyRotatorSystem(config);
	
	// Adding bodies to CUDA system
	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		addBodyToRotationSystem(body->getVertexBuffer(), body->getNumVertices(), config);
	}
}

BodyRotator::~BodyRotator(){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "BodyRotator.cpp\t\tFinalizing" << std::endl;
	}
	
	finalizeBodyRotatorSystem(config);
}

void BodyRotator::simulateRotation(void){
	double theta;
	double phi;
	glm::dvec3 center;
	
	// Prepare the body rotation
	prepareBodyRotation(config);

	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		
		center = body->getCenter();
		theta = body->getRotationSpeed()*dt;
		phi = body->getInclination();
		
		// Rotation matrix
		glm::dmat4 m1 = glm::translate(glm::dmat4(1.0), -center);
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -phi, glm::dvec3(0, 0, 1));
		glm::dmat4 m3 = glm::rotate(glm::dmat4(1.0), theta, glm::dvec3(0, 1, 0));
		glm::dmat4 m4 = glm::rotate(glm::dmat4(1.0), phi, glm::dvec3(0, 0, 1));
		glm::dmat4 m5 = glm::translate(glm::dmat4(1.0), center);		
		glm::dmat4 mat = m5*m4*m3*m2*m1;
		
		rotateBody(i, &mat[0][0], config);
	}
	
	// Ending the rotation
	endBodyRotation(config);
}
