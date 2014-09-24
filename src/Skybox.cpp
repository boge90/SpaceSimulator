#include "../include/Skybox.hpp"
#include <iostream>

Skybox::Skybox(Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){			
		std::cout << "Skybox.cpp\t\t\tInitializing" << std::endl;
	}
	
	// Shader
	shader = new Shader("src/shaders/skyboxVertex.glsl", "src/shaders/skyboxFragment.glsl", config);
}

Skybox::~Skybox(){
	if((debugLevel & 0x10) == 16){		
		std::cout << "Skybox.cpp\t\t\tFinalizing" << std::endl;
	}
}

void Skybox::render(const GLfloat *mvp){
	std::cout << "Skybox.cpp\t\t\tRender" << std::endl;
}
