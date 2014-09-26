#include "../include/Hud.hpp"
#include "../include/MainHudPage.hpp"
#include "../include/BodyHudPage.hpp"

/// TEST
#include "../include/BMP.hpp"
#include "../include/BmpService.hpp"

#include <iostream>

HUD::HUD(GLFWwindow *window, Simulator *simulator, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "HUD.cpp\t\t\tInitializing" << std::endl;
	}
	
	// Init
	glfwGetWindowSize(window, &width, &height);
	this->window = window;
	this->visible = false;
	this->stride = width*3;
	this->pixels = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
	this->drawService = new DrawService(width, height, pixels, config);
	this->pages = new std::vector<HudPage*>();
	this->activePage = 0;
	
	// Creating GUI elements
	left = new View(10, height/2, 50, 50, config);
	right = new View(width-60, height/2, 50, 50, config);
	
	// Adding Click listeners for left and right
	left->addViewClickedAction(this);
	right->addViewClickedAction(this);
	
	// Adding main HUD page
	pages->push_back(new MainHudPage(10+50+10, 0, width-140, height-1, simulator, config));
	
	// Adding Body HUD pages
	std::vector<Body*> *bodies = simulator->getBodies();
	for(size_t i=0; i<bodies->size(); i++){
		pages->push_back(new BodyHudPage(10+50+10, 0, width-140, height-1, i+2, (*bodies)[i], config));
	}
	
	// Texture
	float borderColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	// Buffering texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	
	// Shader
	shader = new Shader("src/shaders/hudVertex.glsl", "src/shaders/hudFragment.glsl", config);
	
	// Generating vertex buffer
	float z = 0.f;
	float vertices[] = {-1.f, 1.f, z,		// Top left
						-1.f, -1.f, z,		// Bottom left
						1.f, 1.f, z,		// Top right
						1.f, -1.f, z,		// Bottom right
						1.f, 1.f, z,		// Top right
						-1.f, -1.f, z};		// Bottom left
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_DYNAMIC_DRAW);
						
	// Generating texture coordinate buffer						
	float texCords[] = {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 1.f}; // Top Down
	glGenBuffers(1, &texCordsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCordsBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCords), &texCords, GL_DYNAMIC_DRAW);
}

HUD::~HUD(){
	if((debugLevel & 0x10) == 16){		
		std::cout << "HUD.cpp\t\t\tFinalizing" << std::endl;
	}
	
	// Freeing HUD pages
	for(size_t i=0; i<pages->size(); i++){
		delete (*pages)[i];
	}
	
	delete left;
	delete right;
	
	glDeleteTextures(1, &tex);
}

void HUD::update(void){
	// Draw page toggle buttons
	left->draw(drawService);
	right->draw(drawService);
	
	// Draw active page
	(*pages)[activePage]->draw(drawService);
	
	// Transferring the pixels
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
}

void HUD::render(void){
	if(visible){			
		// Activating texture
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		update();
		
		// Binding the Body shader
		shader->bind();
	
		// Get a handle for our "MVP" uniform.
		glm::mat4 mvp = glm::lookAt(glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
		GLuint mvpMatrixId = glGetUniformLocation(shader->getID(), "MVP");
		glUniformMatrix4fv(mvpMatrixId, 1, GL_FALSE, &mvp[0][0]);

		// Binding vertex VBO
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);

		// Binding color VBO
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, texCordsBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float)*2, 0);

		// Draw
		glDrawArrays(GL_TRIANGLES, 0, 6);
	
		// Disabling buffers
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void HUD::toggleVisibility(void){
	this->visible = !visible;
}

bool HUD::isVisible(void){
	return visible;
}

void HUD::hudClicked(int button, int action, int x, int y){
	if(visible && action == GLFW_PRESS){
		if(right->isInside(x, y)){
			right->clicked(button, action);
		}else if(left->isInside(x, y)){
			left->clicked(button, action);
		}
		
		Layout *layout = (*pages)[activePage];
		hudClickedRecur(button, action, x, y, layout);
	}
}

void HUD::hudClickedRecur(int button, int action, int x, int y, Layout *layout){
	std::vector<View*> *children = layout->getChildren();
	
	for(size_t i=0; i<children->size(); i++){
		View *child = (*children)[i];
		
		Layout *cast = dynamic_cast<Layout*>(child);
		if(cast == NULL){
			if(child->isInside(x, y)){
				child->clicked(button, action);
			}
		}else{
			hudClickedRecur(button, action, x, y, cast);
		}
	}
}

void HUD::onClick(View *view, int button, int action){
	if(view == left && activePage > 0){
		activePage--;
	}else if(view == right && activePage < pages->size()-1){
		activePage++;
	}
	
	drawService->fill(0, 0, 0);
}
