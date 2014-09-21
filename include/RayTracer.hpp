#ifndef RAY_TRACER_H
#define RAY_TRACER_H

class RayTracer;

#include "../include/Body.hpp"
#include <vector>

class RayTracer{
	private:
		std::vector<Body*> *bodies;
	public:
		/**
		* Constructs the RayTracer system
		**/
		RayTracer(std::vector<Body*> *bodies);
		
		/**
		* Finalizes the RayTracer system
		**/
		~RayTracer();
		
		/**
		* Simulates the rays for the entire system
		**/
		void simulateRays(void);
};

#endif
