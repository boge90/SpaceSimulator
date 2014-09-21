#ifndef NBODY_H
#define NBODY_H

#include <vector>
#include "../include/Body.hpp"

class Nbody{
	private:
		std::vector<Body*> *bodies;
		double G;
		double dt;
	
	public:
		/**
		*
		**/
		Nbody(std::vector<Body*> *bodies, double dt);
		
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
