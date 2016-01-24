#include "../include/Skybox.hpp"
#include "../include/BmpService.hpp"
#include <iostream>
#include <float.h>

Skybox::Skybox(Config *config){
	// Debug
	this->config = config;
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){			
		std::cout << "Skybox.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->intensity = 1.f;

	// Side vertices
	double val = 1.0;
	double vertices1[] = {-val, val, -val, -val, -val, -val, val, -val, -val, val, -val, -val, val, val, -val, -val, val, -val};
	double vertices2[] = {-val, val, val, -val, -val, val, -val, -val, -val, -val, -val, -val, -val, val, -val, -val, val, val};
	double vertices3[] = {val, val, -val, val, -val, -val, val, -val, val, val, -val, val, val, val, val, val, val, -val};
	double vertices4[] = {-val, val, val, -val, val, -val, val, val, -val, val, val, -val, val, val, val, -val, val, val};
	double vertices5[] = {-val, -val, -val, -val, -val, val, val, -val, val, val, -val, val, val, -val, -val, -val, -val, -val};
	double vertices6[] = {val, val, val, val, -val, val, -val, -val, val, -val, -val, val, -val, val, val, val, val, val};
	
	// Activating texture unit used for all textures	
    glActiveTexture(GL_TEXTURE0);

	// Side 1 (front)
	glGenBuffers(1, &vertexBuffer1);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer1);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices1, GL_STATIC_DRAW);	
    glGenTextures(1, &tex1);
    glBindTexture(GL_TEXTURE_2D, tex1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp1 = BmpService::loadImage("texture/GalaxyTex_PositiveZ.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp1->getData());
	
	// Side 2 (left)
	glGenBuffers(1, &vertexBuffer2);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer2);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices2, GL_STATIC_DRAW);	
    glGenTextures(1, &tex2);
    glBindTexture(GL_TEXTURE_2D, tex2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp2 = BmpService::loadImage("texture/GalaxyTex_NegativeX.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp2->getData());
	
	// Side 3 (right)
	glGenBuffers(1, &vertexBuffer3);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer3);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices3, GL_STATIC_DRAW);	
    glGenTextures(1, &tex3);
    glBindTexture(GL_TEXTURE_2D, tex3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp3 = BmpService::loadImage("texture/GalaxyTex_PositiveX.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp3->getData());
	
	// Side 4 (top)
	glGenBuffers(1, &vertexBuffer4);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer4);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices4, GL_STATIC_DRAW);	
    glGenTextures(1, &tex4);
    glBindTexture(GL_TEXTURE_2D, tex4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp4 = BmpService::loadImage("texture/GalaxyTex_PositiveY.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp4->getData());
	
	// Side 5 (bottom)
	glGenBuffers(1, &vertexBuffer5);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer5);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices5, GL_STATIC_DRAW);
    glGenTextures(1, &tex5);
    glBindTexture(GL_TEXTURE_2D, tex5);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp5 = BmpService::loadImage("texture/GalaxyTex_NegativeY.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp5->getData());
	
	// Side 5 (back)
	glGenBuffers(1, &vertexBuffer6);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer6);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices6, GL_STATIC_DRAW);	
    glGenTextures(1, &tex6);
    glBindTexture(GL_TEXTURE_2D, tex6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    BMP *bmp6 = BmpService::loadImage("texture/GalaxyTex_NegativeZ.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp6->getData());
    
    // Texture coordinates
	float texCords[] = {0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f}; // Top Down
	glGenBuffers(1, &texCordsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCordsBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCords), &texCords, GL_STATIC_DRAW);
	
	// Shader
	shader = new Shader(config);
	
	// Creating shader
	shader->addShader("src/shaders/skyboxVertex.glsl", GL_VERTEX_SHADER);
	shader->addShader("src/shaders/skyboxFragment.glsl", GL_FRAGMENT_SHADER);
	shader->link();
	
	// Freeing host memory
	BmpService::freeImage(bmp1, config);
	BmpService::freeImage(bmp2, config);
	BmpService::freeImage(bmp3, config);
	BmpService::freeImage(bmp4, config);
	BmpService::freeImage(bmp5, config);
	BmpService::freeImage(bmp6, config);
}

Skybox::~Skybox(){
	if((debugLevel & 0x10) == 16){		
		std::cout << "Skybox.cpp\t\tFinalizing" << std::endl;
	}
	
	delete shader;
	
	glDeleteBuffers(1, &vertexBuffer1);
	glDeleteBuffers(1, &vertexBuffer2);
	glDeleteBuffers(1, &vertexBuffer3);
	glDeleteBuffers(1, &vertexBuffer4);
	glDeleteBuffers(1, &vertexBuffer5);
	glDeleteBuffers(1, &vertexBuffer6);
	glDeleteBuffers(1, &texCordsBuffer);
	
	glDeleteTextures(1, &tex1);
	glDeleteTextures(1, &tex2);
	glDeleteTextures(1, &tex3);
	glDeleteTextures(1, &tex4);
	glDeleteTextures(1, &tex5);
	glDeleteTextures(1, &tex6);
}

void Skybox::render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){	
	// Model matrix is identidy matrix, because the skybox should NOT move
	glm::mat4 m1 = glm::rotate(glm::mat4(1), 75.f, glm::vec3(0, 1, 0));
	glm::mat4 m2 = glm::rotate(glm::mat4(1), 70.f, glm::vec3(0, 0, 1));
	glm::mat4 mvp = (*vp) * m2 * m1;

	// Binding the Body shader
	shader->bind();

	// Activating texture unit 0
    glActiveTexture(GL_TEXTURE0);
    
	// Binding texture coordinate buffer
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, texCordsBuffer);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_TRUE, sizeof(float)*2, 0);

	// Get a handle for our "MVP" uniform.
	GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
	glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);
	
	// Setting intensity value
	GLuint intensityId = glGetUniformLocation(shader->getID(), "intensity");
	glUniform1f(intensityId, intensity);
	
	// Drawing side 1
    glBindTexture(GL_TEXTURE_2D, tex1);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer1);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 2
    glBindTexture(GL_TEXTURE_2D, tex2);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer2);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 3
    glBindTexture(GL_TEXTURE_2D, tex3);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer3);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 4
    glBindTexture(GL_TEXTURE_2D, tex4);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer4);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 5
    glBindTexture(GL_TEXTURE_2D, tex5);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer5);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 6
    glBindTexture(GL_TEXTURE_2D, tex6);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer6);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Disable
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Skybox::setIntensity(float intensity){
	this->intensity = intensity;
}
