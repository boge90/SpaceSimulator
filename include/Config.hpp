#ifndef CONFIG_H
#define CONFIG_H

#include <stdlib.h>

class Config{
	private:
		double dt;
		size_t renderingDeviceNumber;
		size_t debugLevel;
		
		// Camera
		bool flipCheck;
		
		// Result
		bool discardResult;
		
		// Visualization
		bool fullscreen;
		
		// MPI
		int size;
		int rank;
	
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
		* Returns the simulation 'delta time' value
		**/
		double getDt(void);
		
		/**
		* Returns the debug levels, each bit represents the ON/OFF switch
		* of a certain debug area
		**/
		size_t getDebugLevel(void);
		
		/**
		* Returns the device number that should be used for rendering
		**/
		size_t getRenderingDeviceNumber(void);
		
		/**
		* Returns the pointer to the mpi size
		**/
		int* getMpiSizePtr(void);
		
		/**
		* Returns the pointer to the mpi rank
		**/
		int* getMpiRankPtr(void);
		
		/**
		* The master is running the visualization, all other processes
		* are running the local body simulations, and will not perform
		* any visualization themself.
		**/
		bool isMaster(void);
		
		/**
		* Returns true if the user has specified fullscreen mode
		**/
		bool isFullscreen(void);
		
		/**
		*
		**/
		bool isDiscardResult(void);
		
		/**
		*
		**/
		bool isFlipCheck(void);
};

#endif
