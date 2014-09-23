#include "../include/BodyRotator.hpp"
#include <iostream>

BodyRotator::BodyRotator(std::vector<Body*> *bodies, double dt){
	// Debug
	std::cout << "BodyRotator.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->dt = dt;
	this->bodies = bodies;
}

BodyRotator::~BodyRotator(){
	// Debug
	std::cout << "BodyRotator.cpp\t\tFinalizing" << std::endl;
}

void BodyRotator::simulateRotation(void){
	double theta = 0.1;
	double phi = 5.5;

	for(size_t i=0; i<bodies->size(); i++){
		Body *body = (*bodies)[i];
		
		GLuint vertexBuffer = body->getVertexBuffer();
		int numVertices = body->getNumVertices();
		glm::dvec3 center = body->getCenter();
		
		glm::dvec3 *vertices = (glm::dvec3*) malloc(numVertices*3*sizeof(double));
		
		// Getting data
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glGetBufferSubData(GL_ARRAY_BUFFER, 0, numVertices*3*sizeof(double), vertices);
		
		// Rotation matrix
		glm::dmat4 m1 = glm::translate(glm::dmat4(1.0), -center);
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -phi, glm::dvec3(0, 0, 1));
		glm::dmat4 m3 = glm::rotate(glm::dmat4(1.0), theta, glm::dvec3(0, 1, 0));
		glm::dmat4 m4 = glm::rotate(glm::dmat4(1.0), phi, glm::dvec3(0, 0, 1));
		glm::dmat4 m5 = glm::translate(glm::dmat4(1.0), center);		
		glm::dmat4 mat = m5*m4*m3*m2*m1;
		
		for(int j=0; j<numVertices; j++){
			glm::dvec3 vertex = vertices[j];
			
			vertex.x = vertex.x*mat[0][0] + vertex.y*mat[1][0] + vertex.z*mat[2][0] + mat[3][0];
			vertex.y = vertex.x*mat[0][1] + vertex.y*mat[1][1] + vertex.z*mat[2][1] + mat[3][1];
			vertex.z = vertex.x*mat[0][2] + vertex.y*mat[1][2] + vertex.z*mat[2][2] + mat[3][2];
			
			vertices[j] = vertex;
		}
		
		// Setting data
    	glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(double), vertices, GL_DYNAMIC_DRAW);
    	free(vertices);
	}
}
