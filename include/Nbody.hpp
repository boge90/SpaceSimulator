#ifndef NBODY_H
#define NBODY_H

#include <vector>
#include "../include/Body.hpp"
#include "../include/Config.hpp"

class Nbody{
	private:
		// Simulation data
		std::vector<Body*> *bodies;
		double G;
		double* dt;
		double *positions;
		double *velocities;
		
		// Misc
		size_t debugLevel;
	
	public:
		/**
		*
		**/
		Nbody(std::vector<Body*> *bodies, Config *config);
		
		/**
		*
		**/
		~Nbody(void);
		
		/**
		* Updates the position of the bodies based on their velocity, then calculates their
		* new velocity based on the forces acting on the bodies
		**/
		void simulateGravity(void);
};

#endif
