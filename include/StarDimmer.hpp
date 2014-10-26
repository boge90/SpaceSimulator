#ifndef STAR_DIMMER_H
#define STAR_DIMMER_H

class StarDimmer;

#include <vector>

#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/Simulator.hpp"

class StarDimmer{
	private:
		// Debug
		size_t debug;
		Config *config;
		
		// Misc
		bool activated;
		
		// Data
		Simulator *simulator;
		std::vector<Body*> *bodies;
	public:
		/**
		*
		**/
		StarDimmer(Simulator *simulator, std::vector<Body*> *bodies, Config *config);
		
		/**
		*
		**/
		~StarDimmer(void);
		
		/**
		*
		**/
		void simulateStarDimming(glm::dvec3 cameraPosition);
		
		/**
		*
		**/
		bool isActivated(void);
		
		/**
		*
		**/
		void setActivated(bool activated);
};

#endif
