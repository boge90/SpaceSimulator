#include "../include/StarDimmer.hpp"
#include <iostream>

StarDimmer::StarDimmer(Simulator *simulator, std::vector<Body*> *bodies, Config *config){
	// Debug
	this->debug = config->getDebugLevel();
	this->config = config;
	if((debug & 0x10) == 16){
		std::cout << "StarDimmer.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->simulator = simulator;
	this->bodies = bodies;
	this->activated = true;
}

StarDimmer::~StarDimmer(void){
	if((debug & 0x10) == 16){
		std::cout << "StarDimmer.cpp\t\tFinalizing" << std::endl;
	}
}

void StarDimmer::simulateStarDimming(glm::dvec3 cameraPosition){
	if(activated){
		float intensity = 0.1f;
		
		for(size_t i=0; i<bodies->size(); i++){
			Body *star = (*bodies)[i];
			
			if(!star->isStar()){continue;}
			
			for(size_t j=0; j<bodies->size(); j++){
				Body *body = (*bodies)[j];
				
				if(body->isStar() || i == j){continue;}
				
				// If camera is behind a body, set intensity to 1.f, break for-loop, and go to next star
				glm::dvec3 l = glm::normalize(star->getCenter() - cameraPosition);
				glm::dvec3 oc = cameraPosition - body->getCenter();

				// Test one				
				double loc = (l.x*oc.x) + (l.y*oc.y) + (l.z*oc.z);
				double ocLengthSquared = glm::length(oc) * glm::length(oc);
				double rSquared = body->getRadius() * body->getRadius();
				double test = loc*loc - ocLengthSquared + rSquared;
				
				// Test two
				double l1 = glm::length(cameraPosition - star->getCenter());
				double l2 = glm::length(body->getCenter() - star->getCenter());
				
				if(test >= 0.0 && l1 > l2){
					intensity = 1.f;
					break;
				}
			}
		}
	
		simulator->getSkybox()->setIntensity(intensity);
	}
}

bool StarDimmer::isActivated(void){
	return activated;
}

void StarDimmer::setActivated(bool activated){
	if(!activated){	
		simulator->getSkybox()->setIntensity(1.f);
	}

	this->activated = activated;
}
