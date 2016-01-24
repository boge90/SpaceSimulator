#ifndef BODY_ROTATOR_H
#define BODY_ROTATOR_H

#include "../include/Body.hpp"
#include "../include/Config.hpp"

#include <vector>

class BodyRotator{
	private:
		// Data
		std::vector<Body*> *bodies;
		double *dt;
		
		// Misc
		Config *config;
		size_t debugLevel;
		
	public:
		/**
		*
		**/
		BodyRotator(std::vector<Body*> *bodies, Config *config);
		
		/**
		*
		**/
		~BodyRotator();
		
		/**
		*
		**/
		void simulateRotation(void);
};

#endif
