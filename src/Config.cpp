#include "../include/Config.hpp"
#include <string.h>

Config::Config(int argc, char **args){
	// Setting defult values
	this->renderingDeviceNumber = 0;
	this->dt = 10.0;
	this->bodyVertexDepth = 4;
	this->debugLevel = 0; // (Main loop), (Memory), (GUI), (Init and Finalize), (Init++ and Finalize++), (CUDA Launch), (Render object), (Calculations)
	
	// Reading program input parameters
	for(int i=0; i<argc; i++){
		if(strcmp(args[i], "--dt") == 0){
			dt = strtod(args[++i], NULL);
		}
		if(strcmp(args[i], "--debug") == 0){
			debugLevel = strtod(args[++i], NULL);
		}
		if(strcmp(args[i], "--bodyVertexDepth") == 0){
			bodyVertexDepth = strtod(args[++i], NULL);
		}
	}
}

Config::~Config(void){
	
}

double Config::getDt(void){
	return dt;
}

size_t Config::getDebugLevel(void){
	return debugLevel;
}

size_t Config::getRenderingDeviceNumber(void){
	return renderingDeviceNumber;
}

size_t Config::getBodyVertexDepth(void){
	return bodyVertexDepth;
}

int* Config::getMpiSizePtr(void){
	return &size;
}

int* Config::getMpiRankPtr(void){
	return &rank;
}

bool Config::isMaster(void){
	return rank == 0;
}
