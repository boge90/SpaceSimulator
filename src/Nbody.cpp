#include "../include/Nbody.hpp"
#include "../include/NbodySystem.cuh"
#include <iostream>

Nbody::Nbody(std::vector<Body*> *bodies, double dt){
	std::cout << "Nbody.cpp\t\tInitializing\n";

	// This
	this->bodies = bodies;
	this->dt = dt;
	this->G = 6.7 * pow(10, -11);
	
	// CUDA System
	initializeNbodySystem();
	
	// Adding vertex buffers to CUDA system
	int size = bodies->size();
	for(int i=0; i<size; i++){
		GLuint buffer = (*bodies)[i]->getVertexBuffer();
		addBodyVertexBuffer(buffer);
	}
}

Nbody::~Nbody(void){
	std::cout << "Nbody.cpp\t\tFinalizing\n";
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
		
		// Updating vertices based on delta vector
		double translation[4*4];
		translation[0] = 1.0;
		translation[1] = 0.0;
		translation[2] = 0.0;
		translation[3] = delta.x;
		
		translation[4] = 0.0;
		translation[5] = 1.0;
		translation[6] = 0.0;
		translation[7] = delta.y;
		
		translation[8] = 0.0;
		translation[9] = 0.0;
		translation[10]= 1.0;
		translation[11]= delta.z;
		
		translation[12]= 0.f;
		translation[13]= 0.f;
		translation[14]= 0.f;
		translation[15]= 1.f;
		
		// Moving all vertices based on translation matrix for body 'i'
		moveBody(i, b1->getNumVertices(), translation, center.x, center.y, center.z, b1->getRadius());
		
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
