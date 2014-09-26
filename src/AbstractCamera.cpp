#include "../include/AbstractCamera.hpp"
#include <float.h>

AbstractCamera::AbstractCamera(Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "AbstractCamera.cpp\tInitializing\n";
	}
	
	// MVP
	projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.001f, 1000000000000.f);
	model = glm::mat4(1.0f);
	
	// Data
	this->position = glm::dvec3(0, 0, 0);
	this->active = false;
	this->shader = new Shader("src/shaders/vertex.glsl", "src/shaders/fragment.glsl", config);
}

AbstractCamera::~AbstractCamera(){
	if((debugLevel & 0x10) == 16){	
		std::cout << "AbstractCamera.cpp\tFinalizing\n";
	}
	
	delete shader;
}

glm::mat4 AbstractCamera::getMVP(void){
	return projection * view * model;
}

void AbstractCamera::render(const GLfloat *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){
	if(active){	
		// TODO ?
	}
}

void AbstractCamera::setActive(bool active){
	this->active = active;
}

glm::dvec3 AbstractCamera::getPosition(void){
	return position;
}

glm::dvec3 AbstractCamera::getDirection(void){
	return direction;
}

glm::dvec3 AbstractCamera::getUp(void){
	return up;
}
