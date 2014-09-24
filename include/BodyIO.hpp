#ifndef BODYIO_H
#define BODYIO_H

#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include <vector>

class BodyIO{
	public:
		/**
		* Function for reading data from disk
		**/
		static void read(double *time, std::vector<Body*> *bodies, Config *config);
		
		/**
		* Function for writing data to disk
		**/
		static void write(double time, std::vector<Body*> *bodies, Config *config);
};

#endif
