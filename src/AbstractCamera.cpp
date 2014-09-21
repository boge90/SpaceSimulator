#include "../include/AbstractCamera.hpp"
#include <cfloat>

AbstractCamera::AbstractCamera(){
	//std::cout << "AbstractCamera.cpp\tInitializing\n";
	
	// MVP
	projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.001f, 1000000000000.f);
	model = glm::mat4(1.0f);
}

AbstractCamera::~AbstractCamera(){
	//std::cout << "AbstractCamera.cpp\tFinalizing\n";
}

glm::mat4 AbstractCamera::getMVP(void){
	return projection * view * model;
}

void AbstractCamera::activated(void){
	std::cout << "AbstractCamera.cpp\tActivated\n";
}
