#ifndef RAY_TRACER_H
#define RAY_TRACER_H

class RayTracer;

#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include <vector>

class RayTracer{
	private:
		// Data
		std::vector<Body*> *bodies;
		
		// Misc
		size_t debugLevel;
		
	public:
		/**
		* Constructs the RayTracer system
		**/
		RayTracer(std::vector<Body*> *bodies, Config *config);
		
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
