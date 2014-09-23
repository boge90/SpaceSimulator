#include "../include/Body.hpp"
#include "../include/BmpService.hpp"
#include <iostream>
#include <math.h>

int Body::bodyNumber = 0;

Body::Body(glm::dvec3 center, glm::dvec3 velocity, glm::vec3 rgb, double radius, double mass, double inclination, double rotationSpeed, bool star){
	// Debug
	std::cout << "Body.cpp\t\tInitializing body " << bodyNumber << " (" << this << ")" << "\n";
	std::cout << "Body.cpp\t\t\tCenter   = " << center.x << ", " << center.y << ", " << center.z << "\n";
	std::cout << "Body.cpp\t\t\tVelocity = " << velocity.x << ", " << velocity.y << ", " << velocity.z << "\n";
	std::cout << "Body.cpp\t\t\tRadius   = " << radius << "\n";
	std::cout << "Body.cpp\t\t\tMass     = " << mass << "\n";
	std::cout << "Body.cpp\t\t\tRotation = " << rotationSpeed << "\n";
	std::cout << "Body.cpp\t\t\tStar     = " << star << "\n";
	
	// Init
	this->bodyNum = bodyNumber++;
	
	this->center = center;
	this->velocity = velocity;
	this->rgb = rgb;
	this->radius = radius;
	this->mass = mass;
	this->inclination = inclination;
	this->rotationSpeed = rotationSpeed;
	this->star = star;
	
	this->wireFrame = false;
	this->force = glm::dvec3(0.0, 0.0, 0.0);
	this->bmp = BmpService::loadImage("texture/earthSmall.bmp");
}

Body::~Body(){
	std::cout << "Body.cpp\t\tFinalizing\n";
	
	// Shader
	delete shader;
	
	// Freeing texture image memory
	BmpService::freeImage(bmp);
	
	// Freeing buffers
	glDeleteBuffers(1, &indexBuffer);
	glDeleteBuffers(1, &colorBuffer);
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteTextures(1, &tex);
}

void Body::init(){
	// Loading shader
	//shader = new Shader("src/shaders/bodyVertex.glsl", "src/shaders/bodyFragment.glsl");
	shader = new Shader("src/shaders/vertex.glsl", "src/shaders/fragment.glsl");
	
	// Calculates vertices and colors
	std::vector<glm::dvec3> *vertices = new std::vector<glm::dvec3>();
	std::vector<glm::vec3> *color = new std::vector<glm::vec3>();
	std::vector<GLuint> *indices = new std::vector<GLuint>();
	
	// Starting vertices
	vertices->push_back(glm::dvec3(center.x, center.y + radius, center.z));
	vertices->push_back(glm::dvec3(center.x, center.y, center.z - radius));
	vertices->push_back(glm::dvec3(center.x + radius, center.y, center.z));
	vertices->push_back(glm::dvec3(center.x, center.y, center.z + radius));
	vertices->push_back(glm::dvec3(center.x - radius, center.y, center.z));
	vertices->push_back(glm::dvec3(center.x, center.y - radius, center.z));
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	
	int depth = 4;
	generateVertices(vertices, color, indices, 0, 2, 1, 0, depth);
	generateVertices(vertices, color, indices, 0, 3, 2, 0, depth);
	generateVertices(vertices, color, indices, 0, 4, 3, 0, depth);
	generateVertices(vertices, color, indices, 0, 1, 4, 0, depth);
	
	generateVertices(vertices, color, indices, 5, 1, 2, 0, depth);
	generateVertices(vertices, color, indices, 5, 2, 3, 0, depth);
	generateVertices(vertices, color, indices, 5, 3, 4, 0, depth);
	generateVertices(vertices, color, indices, 5, 4, 1, 0, depth);
	
	numVertices = vertices->size();
	numIndices = indices->size();
	
	// Normalize vertices into sphere
	for(int i=0; i<numVertices; i++){
		// Calculating direction vector
		glm::dvec3 vertex = (*vertices)[i];
		vertex -= center;
		vertex /= glm::length(vertex);
		
		vertex.x = vertex.x*radius + center.x;
		vertex.y = vertex.y*radius + center.y;
		vertex.z = vertex.z*radius + center.z;
		
		(*vertices)[i] = vertex;
	}
	
	// Creating VBOs for vertices, colors and indices
	glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices*sizeof(GLuint), &(indices->front()), GL_STATIC_DRAW);
	glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(double), &(vertices->front()), GL_DYNAMIC_DRAW);
	glGenBuffers(1, &colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(float), &(color->front()), GL_DYNAMIC_DRAW);
    
    // Texture coordinate buffer
    glGenBuffers(1, &texCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
    std::vector<glm::vec2> *coords = new std::vector<glm::vec2>();
    for(size_t i=0; i<vertices->size(); i++){
    	glm::dvec3 vertex = (*vertices)[i];
    	
    	vertex = center - vertex;
    	vertex = glm::normalize(vertex);
    	
    	float u = 0.5 + ((atan2(vertex.z, vertex.x))/(2*3.1415));
    	float v = 0.5 - (asin(vertex.y)/3.1415);
    	
    	coords->push_back(glm::vec2(u, v));
    }
    glBufferData(GL_ARRAY_BUFFER, coords->size()*2*sizeof(float), coords, GL_STATIC_DRAW);
    
    // Generating texture
    glActiveTexture(GL_TEXTURE1 + bodyNum);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glGenerateMipmap(GL_TEXTURE_2D);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp->getWidth(), bmp->getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp->getData());
    
    // Freeing temporary memory
    free(vertices);
    free(color);
    free(indices);
    free(coords);
    
    std::cout << "Body.cpp\t\tInitialized " << numVertices << " vertices for body " << bodyNum << std::endl;
}

void Body::generateVertices(std::vector<glm::dvec3> *vertices, std::vector<glm::vec3> *colors, std::vector<GLuint> *indices, int i1, int i2, int i3, int currentDepth, int finalDepth){
	if(currentDepth < finalDepth){ // Generate more vertices	
		// Triangle vertices
		glm::dvec3 v1 = (*vertices)[i1];
		glm::dvec3 v2 = (*vertices)[i2];
		glm::dvec3 v3 = (*vertices)[i3];
		
		// Calculating additional vertices
		int u1Idx, u2Idx, u3Idx;
		glm::dvec3 u1 = glm::dvec3(((v2.x-v1.x)/2) + v1.x, ((v2.y-v1.y)/2) + v1.y, ((v2.z-v1.z)/2) + v1.z);
		glm::dvec3 u2 = glm::dvec3(((v3.x-v1.x)/2) + v1.x, ((v3.y-v1.y)/2) + v1.y, ((v3.z-v1.z)/2) + v1.z);
		glm::dvec3 u3 = glm::dvec3(((v3.x-v2.x)/2) + v2.x, ((v3.y-v2.y)/2) + v2.y, ((v3.z-v2.z)/2) + v2.z);
		
		colors->push_back(rgb);
		vertices->push_back(u1);
		u1Idx = vertices->size()-1;
		
		colors->push_back(rgb);
		vertices->push_back(u2);
		u2Idx = vertices->size()-1;
		
		colors->push_back(rgb);
		vertices->push_back(u3);
		u3Idx = vertices->size()-1;
		
		// Recurr
		generateVertices(vertices, colors, indices, u1Idx, i2, u3Idx, (currentDepth+1), finalDepth); // Top
		generateVertices(vertices, colors, indices, u1Idx, u3Idx, u2Idx, (currentDepth+1), finalDepth); // Middle
		generateVertices(vertices, colors, indices, i1, u1Idx, u2Idx, (currentDepth+1), finalDepth); // Lower left
		generateVertices(vertices, colors, indices, u2Idx, u3Idx, i3, (currentDepth+1), finalDepth); // Lower right
	}else{ // Generate indices
		indices->push_back(i1);
		indices->push_back(i2);
		indices->push_back(i3);
	}
}

void Body::render(const GLfloat *mvp){
	// Binding the Body shader
	shader->bind();
	
	// Get a handle for our "MVP" uniform.
	GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
	glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, mvp);

	// Binding vertex VBO
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	
	// Binding color VBO
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	
	// Binding texture coordinate buffer
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_TRUE, sizeof(float)*2, 0);
	
	// Indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    
	// Enable wireframe
	if(wireFrame){
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	
	// Draw
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, (void*)0);
	
	// Disable wireframe
	if(wireFrame){	
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	// Disabling buffers
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

glm::dvec3 Body::getCenter(void){
	return center;
}

glm::dvec3 Body::getVelocity(void){
	return velocity;
}

glm::dvec3 Body::getForce(void){
	return force;
}

glm::vec3 Body::getRGB(void){
	return rgb;
}

double Body::getMass(void){
	return mass;
}

double Body::getRadius(void){
	return radius;
}

void Body::setCenter(glm::dvec3 center){
	this->center = center;
}

void Body::setVelocity(glm::dvec3 velocity){
	this->velocity = velocity;
}

void Body::setForce(glm::dvec3 force){
	this->force = glm::vec3(force);
}

GLuint Body::getVertexBuffer(void){
	return vertexBuffer;
}

GLuint Body::getColorBuffer(void){
	return colorBuffer;
}

int Body::getNumVertices(void){
	return numVertices;
}

void Body::setWireframeMode(bool active){
	this->wireFrame = active;
	
	if(wireFrame){
		std::cout << "Body.cpp\t\tTurning ON wireframe for body " << bodyNum << std::endl;
	}else{
		std::cout << "Body.cpp\t\tTurning OFF wireframe for body " << bodyNum << std::endl;
	}
}

double Body::getInclination(void){
	return inclination;
}
		
double Body::getRotationSpeed(void){
	return rotationSpeed;
}

bool Body::isStar(void){
	return star;
}
