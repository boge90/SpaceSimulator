#include "../include/RayTracer.hpp"
#include "../include/RayTracerSystem.cuh"
#include "iostream"

RayTracer::RayTracer(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "RayTracer.cpp\t\tInitializing\n";
	}
	
	// Init
	this->level = 0;
	this->illuminated = true;
	this->bodies = bodies;
	
	// Adding buffers to Ray tracer CUDA system
	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		
		addBodyToRayTracer(body->getVertexBuffer(), body->getSolarCoverageBuffer(), body->getNumVertices(), body->isStar(), config);
	}
}

RayTracer::~RayTracer(){
	// Debug
	if((debugLevel & 0x10) == 16){
		std::cout << "RayTracer.cpp\t\tFinalizing\n";
	}
}

void RayTracer::setLevel(int level){
	this->level = level;
}

void RayTracer::simulateRays(){
	prepareRaySimulation();
	if(level == 0 && !illuminated){
		simulateRaysLevelOff();
	}else if(level == 1){
		simulateRaysLevelOne();
	}else if(level == 2){
		simulateRaysLevelTwo();
	}
	finalizeRaySimulation();
}

void RayTracer::simulateRaysLevelOff(){
	for(size_t i=0; i<bodies->size(); i++){
		rayTracerIllunimate(i);
	}
	illuminated = true;
}

void RayTracer::simulateRaysLevelOne(){
	for(size_t i=0; i<bodies->size(); i++){
		Body *star = (*bodies)[i];
	
		if(!star->isStar()){continue;}

		// Body i is a star, simulate with other bodies
		for(size_t j=0; j<bodies->size(); j++){
			Body *body = (*bodies)[j];
		
			if(body->isStar() || body->getBodyType() == 2 || i == j){continue;}	// Not simulate rays with other potential stars
		
			// Simulation data
			glm::dvec3 c1 = star->getCenter();
			glm::dvec3 c2 = body->getCenter();
		
			glm::dmat4 m1 = glm::translate(glm::dmat4(1.0), c2);
			glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), (body->getInclination()*180.0)/M_PI, glm::dvec3(0.0, 0.0, 1.0));
			glm::dmat4 m3 = glm::rotate(glm::dmat4(1.0), (body->getRotation()*180.0)/M_PI, glm::dvec3(0.0, 1.0, 0.0));
			glm::dmat4 mat = m1*m2*m3;
		
			rayTracerSimulateRays(i, c1.x, c1.y, c1.z, j, c2.x, c2.y, c2.z, &mat[0][0]);
		}
	}

	illuminated = false;
}

void RayTracer::simulateRaysLevelTwo(){
	illuminated = false;
}
