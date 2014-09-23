#ifndef BODY_ROTATOR_H
#define BODY_ROTATOR_H

#include "../include/Body.hpp"

#include <vector>

class BodyRotator{
	private:
		// Data
		std::vector<Body*> *bodies;
		double dt;
		
	public:
		/**
		*
		**/
		BodyRotator(std::vector<Body*> *bodies, double dt);
		
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
