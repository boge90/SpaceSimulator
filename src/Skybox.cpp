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

	// Activating texture unit used for all textures	
    glActiveTexture(GL_TEXTURE0);

	// Side 1 (front)
	glGenBuffers(1, &vertexBuffer1);
    glGenTextures(1, &tex1);
    glBindTexture(GL_TEXTURE_2D, tex1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp1 = BmpService::loadImage("texture/GalaxyTex_PositiveZ.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp1->getData());
	
	// Side 2 (left)
	glGenBuffers(1, &vertexBuffer2);
    glGenTextures(1, &tex2);
    glBindTexture(GL_TEXTURE_2D, tex2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp2 = BmpService::loadImage("texture/GalaxyTex_NegativeX.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp2->getData());
	
	// Side 3 (right)
	glGenBuffers(1, &vertexBuffer3);
    glGenTextures(1, &tex3);
    glBindTexture(GL_TEXTURE_2D, tex3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp3 = BmpService::loadImage("texture/GalaxyTex_PositiveX.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp3->getData());
	
	// Side 4 (top)
	glGenBuffers(1, &vertexBuffer4);
    glGenTextures(1, &tex4);
    glBindTexture(GL_TEXTURE_2D, tex4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp4 = BmpService::loadImage("texture/GalaxyTex_PositiveY.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp4->getData());
	
	// Side 5 (bottom)
	glGenBuffers(1, &vertexBuffer5);
    glGenTextures(1, &tex5);
    glBindTexture(GL_TEXTURE_2D, tex5);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp5 = BmpService::loadImage("texture/GalaxyTex_NegativeY.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp5->getData());
	
	// Side 5 (back)
	glGenBuffers(1, &vertexBuffer6);
    glGenTextures(1, &tex6);
    glBindTexture(GL_TEXTURE_2D, tex6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    bmp6 = BmpService::loadImage("texture/GalaxyTex_NegativeZ.bmp", config);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp1->getWidth(), bmp1->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, bmp6->getData());
    
    // Texture coordinates
	float texCords[] = {0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f}; // Top Down
	glGenBuffers(1, &texCordsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCordsBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCords), &texCords, GL_STATIC_DRAW);
	
	// Shader
	shader = new Shader("src/shaders/skyboxVertex.glsl", "src/shaders/skyboxFragment.glsl", config);
}

Skybox::~Skybox(){
	if((debugLevel & 0x10) == 16){		
		std::cout << "Skybox.cpp\t\tFinalizing" << std::endl;
	}
	
	delete shader;
	
	BmpService::freeImage(bmp1, config);
	BmpService::freeImage(bmp2, config);
	BmpService::freeImage(bmp3, config);
	BmpService::freeImage(bmp4, config);
	BmpService::freeImage(bmp5, config);
	BmpService::freeImage(bmp6, config);
}

void Skybox::render(const GLfloat *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){	
	// Vars
	double val = 1000000000000000;
	double vertices1[] = {position.x-val, position.y+val, position.z-val, position.x-val, position.y-val, position.z-val, position.x+val, position.y-val, position.z-val, position.x+val, position.y-val, position.z-val, position.x+val, position.y+val, position.z-val, position.x-val, position.y+val, position.z-val};
	
	double vertices2[] = {position.x-val, position.y+val, position.z+val, position.x-val, position.y-val, position.z+val, position.x-val, position.y-val, position.z-val, position.x-val, position.y-val, position.z-val, position.x-val, position.y+val, position.z-val, position.x-val, position.y+val, position.z+val};
	
	double vertices3[] = {position.x+val, position.y+val, position.z-val, position.x+val, position.y-val, position.z-val, position.x+val, position.y-val, position.z+val, position.x+val, position.y-val, position.z+val, position.x+val, position.y+val, position.z+val, position.x+val, position.y+val, position.z-val};
	
	double vertices4[] = {position.x-val, position.y+val, position.z+val, position.x-val, position.y+val, position.z-val, position.x+val, position.y+val, position.z-val, position.x+val, position.y+val, position.z-val, position.x+val, position.y+val, position.z+val, position.x-val, position.y+val, position.z+val};
	
	double vertices5[] = {position.x-val, position.y-val, position.z-val, position.x-val, position.y-val, position.z+val, position.x+val, position.y-val, position.z+val, position.x+val, position.y-val, position.z+val, position.x+val, position.y-val, position.z-val, position.x-val, position.y-val, position.z-val};
	
	double vertices6[] = {position.x+val, position.y+val, position.z+val, position.x+val, position.y-val, position.z+val, position.x-val, position.y-val, position.z+val, position.x-val, position.y-val, position.z+val, position.x-val, position.y+val, position.z+val, position.x+val, position.y+val, position.z+val};

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
	glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, mvp);
	
	// Drawing side 1
    glBindTexture(GL_TEXTURE_2D, tex1);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer1);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices1, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 2
    glBindTexture(GL_TEXTURE_2D, tex2);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer2);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices2, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 3
    glBindTexture(GL_TEXTURE_2D, tex3);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer3);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices3, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 4
    glBindTexture(GL_TEXTURE_2D, tex4);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer4);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices4, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 5
    glBindTexture(GL_TEXTURE_2D, tex5);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer5);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices5, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Drawing side 6
    glBindTexture(GL_TEXTURE_2D, tex6);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer6);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double)*3, 0);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(double), vertices6, GL_DYNAMIC_DRAW);	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	
	// Disable
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
