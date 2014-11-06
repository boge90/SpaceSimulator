#include "../include/AbstractCamera.hpp"

AbstractCamera::AbstractCamera(Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "AbstractCamera.cpp\tInitializing\n";
	}
	
	// Init
	this->fov = 70.f;
	this->position = glm::dvec3(0, 0, 0);
	
	// MVP
	projection = glm::perspectiveFov(float(M_PI)*fov/180.f, 1800.f, 1000.f, 0.001f, 1000000000000.f);
}

AbstractCamera::~AbstractCamera(){
	if((debugLevel & 0x10) == 16){
		std::cout << "AbstractCamera.cpp\tFinalizing\n";
	}
}

glm::mat4 AbstractCamera::getVP(void){
	return projection * view;
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

float AbstractCamera::getFieldOfView(void){
	return fov;
}

void AbstractCamera::setFieldOfView(float fov){
	this->fov = fov;
	this->projection = glm::perspectiveFov(float(M_PI)*fov/180.f, 1800.f, 1000.f, 0.001f, 1000000000000.f);
}

void AbstractCamera::setActive(bool active){
	this->active = active;
}
