#include "../include/Config.hpp"
#include <string.h>

Config::Config(int argc, char **args){
	// Setting defult values
	this->renderingDeviceNumber = 0;
	this->dt = 10.0;
	this->bodyVertexDepth = 4;
	this->debugLevel = 0; // (Main loop), (Memory), (GUI), (Init and Finalize), (Init++ and Finalize++), (CUDA Launch), (Render object), (Calculations)
	this->fullscreen = false;
	this->discardResult = false;
	this->flipCheck = false;
	
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
		if(strcmp(args[i], "--fullscreen") == 0){
			fullscreen = true;
		}
		if(strcmp(args[i], "--discardResult") == 0){
			discardResult = true;
		}
		if(strcmp(args[i], "--flipCheck") == 0){
			flipCheck = true;
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

bool Config::isFullscreen(void){
	return fullscreen;
}

bool Config::isDiscardResult(void){
	return discardResult;
}

bool Config::isFlipCheck(void){
	return flipCheck;
}
