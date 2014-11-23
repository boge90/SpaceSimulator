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
		
		addBodyToRayTracer(body->getVertexBuffer(), body->getSolarCoverageBuffer(), body->isStar(), config);
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
		Body *body = (*bodies)[i];
		rayTracerIllunimate(i, body->getNumVertices());
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
			glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), (body->getInclination()), glm::dvec3(0.0, 0.0, 1.0));
			glm::dmat4 m3 = glm::rotate(glm::dmat4(1.0), (body->getRotation()), glm::dvec3(0.0, 1.0, 0.0));
			glm::dmat4 mat = m1*m2*m3;
		
			rayTracerSimulateRaysOne(i, c1.x, c1.y, c1.z, j, c2.x, c2.y, c2.z, body->getNumVertices(), &mat[0][0]);
		}
	}

	illuminated = false;
}

void RayTracer::simulateRaysLevelTwo(){
	for(size_t i=0; i<bodies->size(); i++){
		Body *b = (*bodies)[i];
		if(!b->isStar()){
			rayTracerUnillunimate(i, b->getNumVertices());
		}
	}

	for(size_t i=0; i<bodies->size(); i++){
		Body *source = (*bodies)[i];

		for(size_t j=0; j<bodies->size(); j++){
			Body *body = (*bodies)[j];
		
			if(i == j){continue;}	// Not simulate rays with self
		
			// Simulation data
			glm::dvec3 c1 = source->getCenter();
			glm::dvec3 c2 = body->getCenter();
		
			// Body translation matrix
			glm::dmat4 m1 = glm::translate(glm::dmat4(1.0), c2);
			glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), (body->getInclination()), glm::dvec3(0.0, 0.0, 1.0));
			glm::dmat4 m3 = glm::rotate(glm::dmat4(1.0), (body->getRotation()), glm::dvec3(0.0, 1.0, 0.0));
			glm::dmat4 mat1 = m1*m2*m3;
		
			// Source translation matrix
			glm::dmat4 mm1 = glm::translate(glm::dmat4(1.0), c1);
			glm::dmat4 mm2 = glm::rotate(glm::dmat4(1.0), (source->getInclination()), glm::dvec3(0.0, 0.0, 1.0));
			glm::dmat4 mm3 = glm::rotate(glm::dmat4(1.0), (source->getRotation()), glm::dvec3(0.0, 1.0, 0.0));
			glm::dmat4 mat2 = mm1*mm2*mm3;
			
			if(source->isStar()){			
				rayTracerSimulateRaysTwo(i, c1.x, c1.y, c1.z, source->getNumVertices(), j, c2.x, c2.y, c2.z, body->getNumVertices(), &mat1[0][0], &mat2[0][0], 1.f);
			}else{
				double dist = glm::length(c2 - c1);
				float intensity = (4.f*powf(10, 7)) / dist;
				
				if(intensity > 0.01){ // Only simulate Body -> body light when they are close
					rayTracerSimulateRaysTwo(i, c1.x, c1.y, c1.z, source->getNumVertices(), j, c2.x, c2.y, c2.z, body->getNumVertices(), &mat1[0][0], &mat2[0][0], intensity);
				}
			}
		}
	}

	illuminated = false;
}
