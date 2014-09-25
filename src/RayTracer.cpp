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

void RayTracer::simulateRays(){
	prepareRaySimulation();

	for(size_t i=0; i<bodies->size(); i++){
		Body *star = (*bodies)[i];
		
		if(!star->isStar()){continue;}
	
		// Body i is a star, simulate with other bodies
		for(size_t j=0; j<bodies->size(); j++){
			Body *body = (*bodies)[j];
			
			if(body->isStar() || i == j){continue;}	// Not simulate rays with other potential stars
			
			// Simulation data
			glm::dvec3 c1 = star->getCenter();
			glm::dvec3 c2 = body->getCenter();
			
			rayTracerSimulateRays(i, c1.x, c1.y, c1.z, j, c2.x, c2.y, c2.z);
		}
	}
	
	finalizeRaySimulation();
}
