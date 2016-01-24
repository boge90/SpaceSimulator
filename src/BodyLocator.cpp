#include "../include/BodyLocator.hpp"
#include <iostream>

BodyLocator::BodyLocator(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->config = config;
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "BodyLocator.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->bodies = bodies;
	this->active = false;
	this->bodyIndex = 0;
	this->shader = new Shader(config);
	
	// Creating shader
	shader->addShader("src/shaders/vertex.glsl", GL_VERTEX_SHADER);
	shader->addShader("src/shaders/fragment.glsl", GL_FRAGMENT_SHADER);
	shader->link();
	
	// Buffers
	glGenBuffers(1, &vertexBuffer);
	
	float colors[] = {1.f, 1.f, 1.f,1.f, 1.f, 1.f};
	glGenBuffers(1, &colorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_DYNAMIC_DRAW);
}
		
BodyLocator::~BodyLocator(){
	if((debugLevel & 0x10) == 16){
		std::cout << "BodyLocator.cpp\t\tFinalizing" << std::endl;
	}
	 
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &colorBuffer);
}
	
void BodyLocator::render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){
	if(active){
		// Binding shader
		shader->bind();
		
		// Translating and rotating body
		glm::mat4 mvp = (*vp) * glm::mat4(1);
	
		// Get a handle for our "MVP" uniform.
		GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
		glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);
		
		// Calculating vertices
		glm::dvec3 vector = (*bodies)[bodyIndex]->getCenter() - position;
		
		vector = glm::normalize(vector)*0.2;
		float vertices[] = {float(direction.x), float(direction.y), float(direction.z), float(direction.x+vector.x), float(direction.y+vector.y), float(direction.z+vector.z)};
		
		// Vertex buffer
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
		
		// Color buffer
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
		
		// Draw
		glDrawArrays(GL_LINES, 0, 6);
		
		// Disable shader attributes
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}
}
	
void BodyLocator::locateBody(size_t bodyIndex){
	if(bodyIndex < 0){
		bodyIndex = 0;
	}else if(bodyIndex >= bodies->size()){
		bodyIndex = bodies->size() - 1;
	}
	this->bodyIndex = bodyIndex;
}

void BodyLocator::setActive(bool active){
	this->active = active;
}
