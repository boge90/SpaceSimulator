#include "../include/Config.hpp"
#include <string.h>
#include <iostream>

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
	for(int i=1; i<argc; i++){
		if(strcmp(args[i], "--dt") == 0){
			dt = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--debug") == 0){
			debugLevel = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--bodyVertexDepth") == 0){
			bodyVertexDepth = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--fullscreen") == 0){
			fullscreen = true;
		}else if(strcmp(args[i], "--discardResult") == 0){
			discardResult = true;
		}else if(strcmp(args[i], "--flipCheck") == 0){
			flipCheck = true;
		}else if(strcmp(args[i], "--help") == 0){
			std::cout << "--dt FLOAT            - Controls the delta time used in the simulation" << std::endl;
			std::cout << "--debug INT           - Controls the debugging level, where each bit in the integer represents a debug switch" << std::endl;
			std::cout << "--bodyVertexDepth INT - Controls the recursion level when creating the vertices for the bodies, higher yields finer mesh" << std::endl;
			std::cout << "--fullscreen          - Turns on fullscreen mode" << std::endl;
			std::cout << "--discardResult       - Turns off the writing back the end result to disk" << std::endl;
			std::cout << "--flipCheck           - Turns on the flip check for the Body camera" << std::endl;
			exit(EXIT_SUCCESS);
		}else{
			std::cout << "Config.cpp\t\tNo options for arg " << args[i] << std::endl;
			exit(EXIT_FAILURE);
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
