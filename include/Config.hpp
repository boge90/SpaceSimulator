#ifndef CONFIG_H
#define CONFIG_H

#include <stdlib.h>

class Config{
	private:
		double dt;
		size_t renderingDeviceNumber;
		size_t debugLevel;
		size_t bodyVertexDepth;
	
	public:
		/**
		* Creates the config object base on the program startup parameters
		**/
		Config(int argc, char **args);
		
		/**
		* Finalizes the config obejct
		**/
		~Config(void);
		
		/**
		* 
		**/
		double getDt(void);
		
		/**
		*
		**/
		size_t getDebugLevel(void);
		
		/**
		*
		**/
		size_t getRenderingDeviceNumber(void);
		
		/**
		*
		**/
		size_t getBodyVertexDepth(void);
};

#endif
