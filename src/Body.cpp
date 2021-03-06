#include "../include/Body.hpp"
#include "../include/BmpService.hpp"
#include <iostream>
#include <math.h>

int Body::bodyNumber = 0;

Body::Body(std::string name, glm::dvec3 center, glm::dvec3 velocity, glm::vec3 rgb, glm::vec3 atmosphereColor, double rotation, double radius, double mass, double inclination, double rotationSpeed, BodyType bodyType, std::string texturePath, Config *config){
	// Debug
	this->config = config;
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "Body.cpp\t\tInitializing body " << bodyNumber << " (" << this << ")" << "\n";
	}
	
	if((debugLevel & 0x8) == 8){	
		std::cout << "Body.cpp\t\t\tName     = " << name << "\n";
		std::cout << "Body.cpp\t\t\tCenter   = " << center.x << ", " << center.y << ", " << center.z << "\n";
		std::cout << "Body.cpp\t\t\tVelocity = " << velocity.x << ", " << velocity.y << ", " << velocity.z << "\n";
		std::cout << "Body.cpp\t\t\tRadius   = " << radius << "\n";
		std::cout << "Body.cpp\t\t\tMass     = " << mass << "\n";
		std::cout << "Body.cpp\t\t\tRotation = " << rotationSpeed << "\n";
		std::cout << "Body.cpp\t\t\tBodyType = " << bodyType << "\n";
	}
	
	// Init
	this->bodyNum = bodyNumber++;
	this->name = name;
	this->center = center;
	this->velocity = velocity;
	this->rgb = rgb;
	this->atmosphereColor = atmosphereColor;
	this->radius = radius;
	this->mass = mass;
	this->inclination = inclination;
	this->rotationSpeed = rotationSpeed;
	this->bodyType = bodyType;
	this->wireFrame = false;
	this->fakeSize = false;
	this->rotation = rotation;
	this->force = glm::dvec3(0.0, 0.0, 0.0);
	this->texturePath = texturePath;
	this->lod = config->getMinBodyLod();
	this->surfaceTemperatureBuffer = 0;
	this->visualization = NORMAL;
}

Body::~Body(){
	if((debugLevel & 0x10) == 16){	
		std::cout << "Body.cpp\t\tFinalizing\n";
	}
	
	// Clean objects
	delete normalShader;
	delete atmosphere;
	
	// Freeing texture image memory
	if(bodyType == STAR || bodyType == PLANET){	
		glDeleteTextures(1, &tex);
		glDeleteBuffers(1, &texCoordBuffer);
	}
	
	// Freeing buffers
	glDeleteBuffers(1, &indexBuffer);
	glDeleteBuffers(1, &colorBuffer);
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &solarCoverBuffer);
}

void Body::init(){
	// Atmosphere
	atmosphere = new Atmosphere(this, config);

	// Loading shaders
	if(bodyType == STAR || bodyType == PLANET){
		normalShader = new Shader(config);
		normalShader->addShader("src/shaders/bodyVertex.glsl", GL_VERTEX_SHADER);
		normalShader->addShader("src/shaders/bodyFragment.glsl", GL_FRAGMENT_SHADER);
	}else{	
		normalShader = new Shader(config);
		normalShader->addShader("src/shaders/cometVertex.glsl", GL_VERTEX_SHADER);
		normalShader->addShader("src/shaders/cometFragment.glsl", GL_FRAGMENT_SHADER);
	}
	
	normalShader->link();
	
	temperatureShader = new Shader(config);
	temperatureShader->addShader("src/shaders/bodyTemperatureVertex.glsl", GL_VERTEX_SHADER);
	temperatureShader->addShader("src/shaders/bodyTemperatureFragment.glsl", GL_FRAGMENT_SHADER);
	temperatureShader->link();
	
	// Generating buffers
	glGenBuffers(1, &colorBuffer);
	glGenBuffers(1, &indexBuffer);
	glGenBuffers(1, &vertexBuffer);
	glGenBuffers(1, &texCoordBuffer);
    glGenBuffers(1, &solarCoverBuffer);
	glGenBuffers(1, &surfaceTemperatureBuffer);
	
	// Calculating vertices
	generateVertices(lod);
    
    // Texture coordinate buffer
	if(bodyType == STAR || bodyType == PLANET){
		// Loading texture	
		BMP *bmp = BmpService::loadImage(texturePath.c_str(), config);	
		
		// Generating texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp->getWidth(), bmp->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp->getData());
		BmpService::freeImage(bmp, config);
		glGenerateMipmap(GL_TEXTURE_2D);
		
		if((debugLevel & 0x40) == 64){
			long mem = (numVertices*3*sizeof(double)) + numVertices*3*sizeof(float) + numIndices*sizeof(GLuint) + (numVertices*2*sizeof(float)) + (numVertices*sizeof(float));
		
			if(mem/(1024*1024) > 0){ // MiB		
				std::cout << "Body.cpp\t\tMemory usage for body " << bodyNum << " is " << (mem/(1024*1024)) << " MiB" << std::endl;
			}else if(mem/1024 > 0){ // KiB
				std::cout << "Body.cpp\t\tMemory usage for body " << bodyNum << " is " << (mem/(1024)) << " KiB" << std::endl;
			}else{
				std::cout << "Body.cpp\t\tMemory usage for body " << bodyNum << " is " << (mem) << " Bytes" << std::endl;
			}
		}
	}
	
	// Surface temperatur buffer
	float *temperatur = (float*) malloc(numVertices*sizeof(float));
	for(size_t i=0; i<numVertices; i++){
		temperatur[i] = 0.f;
	}
    glBindBuffer(GL_ARRAY_BUFFER, surfaceTemperatureBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*sizeof(float), temperatur, GL_DYNAMIC_DRAW);
	
    
	if((debugLevel & 0x8) == 8){	
	    std::cout << "Body.cpp\t\tInitialized " << numVertices << " vertices for body " << bodyNum << std::endl;
	}
}

static inline float clamp(float in, float min, float max)
{
	if(in < min){return min;}
	else if(in > max){return max;}
	else{return in;}
}

void Body::generateVertices(int depth){
	// Calculates vertices and colors
	std::vector<glm::vec3> *vertices = new std::vector<glm::vec3>();
	std::vector<glm::vec3> *color = new std::vector<glm::vec3>();
	std::vector<GLuint> *indices = new std::vector<GLuint>();
	
	// Starting vertices
	vertices->push_back(glm::vec3(0.f, 1.f, 0.f));
	vertices->push_back(glm::vec3(0.f, 0.f, -1.f));
	vertices->push_back(glm::vec3(1.f, 0.f, 0.f));
	vertices->push_back(glm::vec3(0.f, 0.f, 1.f));
	vertices->push_back(glm::vec3(-1.f, 0.f, 0.f));
	vertices->push_back(glm::vec3(0.f, -1.f, 0.f));
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	color->push_back(rgb);
	
	calculateVertices(vertices, color, indices, 0, 2, 1, 0, depth);
	calculateVertices(vertices, color, indices, 0, 3, 2, 0, depth);
	calculateVertices(vertices, color, indices, 0, 4, 3, 0, depth);
	calculateVertices(vertices, color, indices, 0, 1, 4, 0, depth);
	
	calculateVertices(vertices, color, indices, 5, 1, 2, 0, depth);
	calculateVertices(vertices, color, indices, 5, 2, 3, 0, depth);
	calculateVertices(vertices, color, indices, 5, 3, 4, 0, depth);
	calculateVertices(vertices, color, indices, 5, 4, 1, 0, depth);
	
	numVertices = vertices->size();
	numIndices = indices->size();

	// Normalize vertices into sphere
	for(size_t i=0; i<numVertices; i++){
		// Calculating direction vector
		glm::vec3 vertex = (*vertices)[i];
		vertex /= glm::length(vertex);
		
		(*vertices)[i] = vertex;
	}
	
	// Creating VBOs for vertices, colors and indices	
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices*sizeof(GLuint), &(indices->front()), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(float), &(vertices->front()), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(float), &(color->front()), GL_DYNAMIC_DRAW);
    
    
    // Solar coverage buffer
    float *coverage = (float*) malloc(numVertices*sizeof(float));
	for(size_t i=0; i<numVertices; i++){ // Need to set the values of stars, since they are not updated by the RayTracing
		coverage[i] = 1.f;
	}
    glBindBuffer(GL_ARRAY_BUFFER, solarCoverBuffer);
    glBufferData(GL_ARRAY_BUFFER, numVertices*sizeof(float), coverage, GL_DYNAMIC_DRAW);
    
	std::vector<glm::vec2> *coords = new std::vector<glm::vec2>();

	for(size_t i=0; i<vertices->size(); i++){
		glm::vec3 vertex = (*vertices)[i];
		vertex = glm::normalize(vertex);
		
		vertex = -vertex;
		
		float u = 0.5 - ((atan2(vertex.z, vertex.x))/(2.f*M_PI));
		float v = 0.5 - (asin(vertex.y)/M_PI);
		
		u = clamp(u, 0.f, 1.f);
		v = clamp(v, 0.f, 1.f);

		assert(u >= 0 && u <= 1);
		assert(v >= 0 && v <= 1);
		
		coords->push_back(glm::vec2(u, v));
	}
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
	glBufferData(GL_ARRAY_BUFFER, coords->size()*2*sizeof(float), &(coords->front()), GL_DYNAMIC_DRAW);
    
    // Freeing temporary memory
    free(vertices);
    free(color);
    free(coverage);
    free(indices);		
    free(coords);
}

void Body::calculateVertices(std::vector<glm::vec3> *vertices, std::vector<glm::vec3> *colors, std::vector<GLuint> *indices, int i1, int i2, int i3, int currentDepth, int finalDepth){
	if(currentDepth < finalDepth){ // Generate more vertices	
		// Triangle vertices
		glm::vec3 v1 = (*vertices)[i1];
		glm::vec3 v2 = (*vertices)[i2];
		glm::vec3 v3 = (*vertices)[i3];
		
		// Calculating additional vertices
		int u1Idx, u2Idx, u3Idx;
		glm::vec3 u1 = glm::vec3(((v2.x-v1.x)/2) + v1.x, ((v2.y-v1.y)/2) + v1.y, ((v2.z-v1.z)/2) + v1.z);
		glm::vec3 u2 = glm::vec3(((v3.x-v1.x)/2) + v1.x, ((v3.y-v1.y)/2) + v1.y, ((v3.z-v1.z)/2) + v1.z);
		glm::vec3 u3 = glm::vec3(((v3.x-v2.x)/2) + v2.x, ((v3.y-v2.y)/2) + v2.y, ((v3.z-v2.z)/2) + v2.z);
		
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
		calculateVertices(vertices, colors, indices, u1Idx, i2, u3Idx, (currentDepth+1), finalDepth); // Top
		calculateVertices(vertices, colors, indices, u1Idx, u3Idx, u2Idx, (currentDepth+1), finalDepth); // Middle
		calculateVertices(vertices, colors, indices, i1, u1Idx, u2Idx, (currentDepth+1), finalDepth); // Lower left
		calculateVertices(vertices, colors, indices, u2Idx, u3Idx, i3, (currentDepth+1), finalDepth); // Lower right
	}else{ // Generate indices
		indices->push_back(i1);
		indices->push_back(i2);
		indices->push_back(i3);
	}
}

void Body::render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){
	// Size check
	//if(2*asin(radius/glm::length(center - position)) < 0.25*M_PI/180.0 && !fakeSize){
	//	return;
	//}

	// FakeSize
	float scale = radius;
	if(fakeSize){
		scale = (glm::length(center - position))/25.f;
	}
	
	// Render atmosphere
	atmosphere->render(vp, position, direction, up);
	
	// Translating, scaling and rotating body
	glm::mat4 mvp = (*vp) * glm::translate(glm::mat4(1), glm::vec3(center - position));
	mvp = mvp * glm::rotate(glm::mat4(1.f), float(inclination), glm::vec3(0, 0, 1));
	mvp = mvp * glm::rotate(glm::mat4(1.f), float(rotation), glm::vec3(0, 1, 0));
	mvp = mvp * glm::scale(glm::mat4(1.f), glm::vec3(scale, scale, scale));
	
	if ( visualization == NORMAL ){	
		// Binding the Body shader
		normalShader->bind();
	
		// Activating texture
		if(bodyType == STAR || bodyType == PLANET){
			glBindTexture(GL_TEXTURE_2D, tex);
		}
	
		// Get a handle for our "MVP" uniform.
		GLuint mvpMatrixId = glGetUniformLocation(normalShader->getID(), "MVP");
		glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);

		// Binding vertex VBO
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	
		// Binding color VBO
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	
		// Binding solar coverage VBO
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, solarCoverBuffer);
		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), 0);
	
		// Binding texture coordinate buffer
		if(bodyType == STAR || bodyType == PLANET){
			glEnableVertexAttribArray(3);
			glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
			glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, sizeof(float)*2, 0);
		}
	}else if ( visualization == TEMPERATURE ){
		// Binding the Body shader
		temperatureShader->bind();
		
		// Get a handle for our "MVP" uniform.
		GLuint mvpMatrixId = glGetUniformLocation(normalShader->getID(), "MVP");
		glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);

		// Binding vertex VBO
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);

		// Binding vertex VBO
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, surfaceTemperatureBuffer);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), 0);
	}
	
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
	
	if(bodyType == STAR || bodyType == PLANET){	
		glDisableVertexAttribArray(3);
	}
	
	// Error check
	int error = glGetError();
	if(error != 0){
		std::cout << "Body.cpp\t\tOpenGl error " << error << std::endl;
	}
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

glm::vec3 Body::getAtmosphereColor(void){
	return atmosphereColor;
}

double Body::getMass(void){
	return mass;
}

double Body::getRadius(void){
	return radius;
}

void Body::setRotation(double rotation){
	this->rotation = rotation;
}

double Body::getRotation(void){
	return rotation;
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

size_t Body::getLOD(void){
	return lod;
}
		
void Body::setLOD(size_t lod){
	this->lod = lod;
}

GLuint Body::getVertexBuffer(void){
	return vertexBuffer;
}

GLuint Body::getColorBuffer(void){
	return colorBuffer;
}

GLuint Body::getSolarCoverageBuffer(void){
	return solarCoverBuffer;
}

GLuint Body::getVertexIndexBuffer(void){
	return indexBuffer;
}

GLuint Body::getSurfaceTemperatureBuffer(void){
	return surfaceTemperatureBuffer;
}

size_t Body::getNumIndices(void){
	return numIndices;
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

void Body::setVisualizationType(Visualization type){
	switch (type){
		case NORMAL:
			std::cout << "Body.cpp\t\tChanging visualization to NORMAL for " << name << std::endl;
			break;
		case TEMPERATURE:
			std::cout << "Body.cpp\t\tChanging visualization to TEMPERATURE for " << name << std::endl;
			break;
	}
	this->visualization = type;
}

void Body::setFakeSize(bool active){
	this->fakeSize = active;
	
	if(fakeSize){
		std::cout << "Body.cpp\t\tTurning ON fakeSize for body " << bodyNum << std::endl;
	}else{
		std::cout << "Body.cpp\t\tTurning OFF fakeSize for body " << bodyNum << std::endl;
	}
}

double Body::getInclination(void){
	return inclination;
}
		
double Body::getRotationSpeed(void){
	return rotationSpeed;
}

bool Body::isStar(void){
	return bodyType == STAR;
}

BodyType Body::getBodyType(void){
	return bodyType;
}

std::string* Body::getTexturePath(void){
	return &texturePath;
}

std::string* Body::getName(void){
	return &name;
}
