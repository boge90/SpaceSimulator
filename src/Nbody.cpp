#include "../include/Nbody.hpp"
#include <iostream>

Nbody::Nbody(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "Nbody.cpp\t\tInitializing\n";
	}

	// This
	this->dt = config->getDt();
	this->bodies = bodies;
	this->G = 6.7 * pow(10, -11);
}

Nbody::~Nbody(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "Nbody.cpp\t\tFinalizing\n";
	}
}

void Nbody::simulateGravity(void){
	// Calculate force
	int size = bodies->size();
	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			if(i != j){
				// Simulate
				Body *b1 = (*bodies)[i];
				Body *b2 = (*bodies)[j];
				
				double dist_x = b2->getCenter().x - b1->getCenter().x;
				double dist_y = b2->getCenter().y - b1->getCenter().y;
				double dist_z = b2->getCenter().z - b1->getCenter().z;

				double abs_dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
				double dist_cubed = abs_dist*abs_dist*abs_dist;

				glm::dvec3 force = b1->getForce();
				force.x += (G * b1->getMass() * b2->getMass())/dist_cubed * dist_x;
				force.y += (G * b1->getMass() * b2->getMass())/dist_cubed * dist_y;
				force.z += (G * b1->getMass() * b2->getMass())/dist_cubed * dist_z;
				b1->setForce(force);
			}
		}
	}
	
	
	// Update position
	for(int i=0; i<size; i++){
		// Body
		Body *b1 = (*bodies)[i];

		// Retrive calculated force
		glm::dvec3 force = b1->getForce();
		
		// Using current center and delta to generate master matrix
		glm::dvec3 center = b1->getCenter();
		glm::dvec3 delta = dt*b1->getVelocity();
		
		// The below asserts will fail when TOO low DT is being used, causing center + delta to NOT change
		assert(center.x != (center.x+delta.x) || delta.x == 0);
		assert(center.y != (center.y+delta.y) || delta.y == 0);
		assert(center.z != (center.z+delta.z) || delta.z == 0);
		
		// Updating new center
		center += delta;
		b1->setCenter(center);
		
		// Updating new velocity
		glm::dvec3 velocity = b1->getVelocity();
		velocity += dt * force/b1->getMass();
		b1->setVelocity(velocity);
		
		// Reset force for next iteration
		b1->setForce(glm::dvec3(0.0, 0.0, 0.0));
	}
}
