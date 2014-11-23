#ifndef BODY_LEVEL_OF_DETAIL_H
#define BODY_LEVEL_OF_DETAIL_H

#include <vector>

#include "../include/Body.hpp"
#include "../include/Config.hpp"

class BodyLevelOfDetail{
	private:
		// Debug
		size_t debugLevel;
		Config *config;
	
		// Data
		std::vector<Body*> *bodies;
	public:
		/**
		* Creates the LOD updater
		**/
		BodyLevelOfDetail(std::vector<Body*> *bodies, Config *config);
		
		/**
		* Destroys the LOD updater
		**/
		~BodyLevelOfDetail();
		
		/**
		* Updates the LOD for all the bodies IFF required
		**/
		void update(glm::dvec3 cameraPosition);
};

#endif
