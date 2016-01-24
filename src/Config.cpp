#include "../include/Config.hpp"
#include <string.h>
#include <iostream>

Config::Config(int argc, char **args){
	// Setting defult values
	this->renderingDeviceNumber = 0;
	this->dt = 10.0;
	this->debugLevel = 0; // (Main loop), (Memory), (GUI), (Init and Finalize), (Init++ and Finalize++), (CUDA Launch), (Render object), (Calculations)
	this->flipCheck = false;
	this->fullscreen = false;
	this->discardResult = false;
	this->maxBodyLod = 7;
	this->minBodyLod = 1;
	this->mouseSpeed = 1.0;
	this->combinedBodyHudPages = true;
	
	// Reading program input parameters
	for(int i=1; i<argc; i++){
		if(strcmp(args[i], "--dt") == 0){
			dt = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--debug") == 0){
			debugLevel = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--maxBodyLod") == 0){
			maxBodyLod = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--minBodyLod") == 0){
			minBodyLod = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--mouse") == 0){
			mouseSpeed = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--fullscreen") == 0){
			fullscreen = true;
		}else if(strcmp(args[i], "--discardResult") == 0){
			discardResult = true;
		}else if(strcmp(args[i], "--flipCheck") == 0){
			flipCheck = true;
		}else if(strcmp(args[i], "--combinedBodyHudPages") == 0){
			combinedBodyHudPages = strtod(args[++i], NULL);
		}else if(strcmp(args[i], "--help") == 0){
			std::cout << "--dt FLOAT                   - Controls the delta time used in the simulation" << std::endl;
			std::cout << "--debug INT                  - Controls the debugging level, where each bit in the integer represents a debug switch" << std::endl;
			std::cout << "--maxBodyLod INT             - Controls the MAX recursion level when creating the vertices for the bodies, higher yields finer mesh" << std::endl;
			std::cout << "--minBodyLod INT             - Controls the MIN recursion level when creating the vertices for the bodies, lower yields coarser mesh" << std::endl;
			std::cout << "--fullscreen                 - Turns on fullscreen mode" << std::endl;
			std::cout << "--discardResult              - Turns off the writing back the end result to disk" << std::endl;
			std::cout << "--flipCheck                  - Turns on the flip check for the Body camera" << std::endl;
			std::cout << "--mouse FLOAT                - Change mouse sensitivity (Default 1.0)" << std::endl;
			std::cout << "--combinedBodyHudPages BOOL  - Specify if you want a combined HUD page for all bodies (Default 1)" << std::endl;
			exit(EXIT_SUCCESS);
		}else{
			std::cout << "Config.cpp\t\tNo options for arg " << args[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Config.cpp\t\tdt = " << dt << std::endl;
	std::cout << "Config.cpp\t\tFlip Check = " << flipCheck << std::endl;
	std::cout << "Config.cpp\t\tFullscreen = " << fullscreen << std::endl;
	std::cout << "Config.cpp\t\tDiscard Result = " << discardResult << std::endl;
	std::cout << "Config.cpp\t\tMax LOD = " << maxBodyLod << std::endl;
	std::cout << "Config.cpp\t\tMin LOD = " << minBodyLod << std::endl;
	std::cout << "Config.cpp\t\tMouse Sensitivity = " << mouseSpeed << std::endl;
}

Config::~Config(void){
	
}

double* Config::getDt(void){
	return &dt;
}

size_t Config::getDebugLevel(void){
	return debugLevel;
}

double Config::getMouseSpeed(void){
	return mouseSpeed;
}

size_t Config::getMaxBodyLod(void){
	return maxBodyLod;
}

size_t Config::getMinBodyLod(void){
	return minBodyLod;
}

size_t Config::getRenderingDeviceNumber(void){
	return renderingDeviceNumber;
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

bool Config::isCombinedBodyHudPages(void)
{
	return combinedBodyHudPages;
}
