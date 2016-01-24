#include "../include/Atmosphere.hpp"

Atmosphere::Atmosphere(Body *body, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	this->config = config;
	if((debugLevel & 0x10) == 16){
		std::cout << "Atmosphere.cpp\t\tInitializing for body " << body << std::endl;
	}
	
	// Init
	this->body = body;
	this->shader = new Shader(config);
	
	// Creating shader
	shader->addShader("src/shaders/atmosphereVertex.glsl", GL_VERTEX_SHADER);
	shader->addShader("src/shaders/atmosphereFragment.glsl", GL_FRAGMENT_SHADER);
	shader->link();
}

Atmosphere::~Atmosphere(){
	if((debugLevel & 0x10) == 16){
		std::cout << "Atmosphere.cpp\t\tFinalizing" << std::endl;
	}
	
	// Cleanup
	delete shader;
}

void Atmosphere::render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){
	// Binding the Body shader
	shader->bind();

	// Misc	
	float distanceToGround = glm::length(position - body->getCenter()) - body->getRadius();
	float maxAtmosphereHeight = (body->getRadius() * 1.015696123f) - body->getRadius();

	// Setting uniforms
	double scale = body->getRadius();
	glm::mat4 mvp = (*vp) * glm::translate(glm::mat4(1), glm::vec3(body->getCenter() - position));
	mvp = mvp * glm::scale(glm::mat4(1.f), glm::vec3((distanceToGround/maxAtmosphereHeight) + 1.01f));
	mvp = mvp * glm::scale(glm::mat4(1.f), glm::vec3(scale, scale, scale));
	mvp = mvp * glm::rotate(glm::mat4(1.f), float(body->getInclination()), glm::vec3(0, 0, 1));
	mvp = mvp * glm::rotate(glm::mat4(1.f), float(body->getRotation()), glm::vec3(0, 1, 0));
	
	GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
	glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);
	
	
	float intensity = (1.f - (distanceToGround/maxAtmosphereHeight)) * 0.95f;
	if(intensity > 0.f){
		glm::vec3 color = body->getAtmosphereColor() * intensity;
	
		GLuint colorId = glGetUniformLocation(shader->getID(), "in_color");
		glUniform3f(colorId, color.x, color.y, color.z);
	
		// Binding vertex buffer
		GLuint vertexId = glGetAttribLocation(shader->getID(), "in_vertex");
		glEnableVertexAttribArray(vertexId);
		glBindBuffer(GL_ARRAY_BUFFER, body->getVertexBuffer());
		glVertexAttribPointer(vertexId, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
		
		// Binding solar coverage buffer
		GLuint coverageId = glGetAttribLocation(shader->getID(), "in_coverage");
		glEnableVertexAttribArray(coverageId);
		glBindBuffer(GL_ARRAY_BUFFER, body->getSolarCoverageBuffer());
		glVertexAttribPointer(coverageId, 1, GL_FLOAT, GL_FALSE, sizeof(float), 0);
	
		// Indices
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, body->getVertexIndexBuffer());

		// Culling	( Atmoshere is visible from space AND body )
		glDisable(GL_CULL_FACE);
	
		// Draw
		glDrawElements(GL_TRIANGLES, body->getNumIndices(), GL_UNSIGNED_INT, (void*)0);

		// Culling	
		glEnable(GL_CULL_FACE);

		// Disabling buffers
		glDisableVertexAttribArray(0);
	
		// Error check
		int error = glGetError();
		if(error != 0){
			std::cout << "Atmosphere.cpp\t\tOpenGl error " << error << std::endl;
		}
	}
}
