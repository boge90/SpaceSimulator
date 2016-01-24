#include "../include/AbstractCamera.hpp"

AbstractCamera::AbstractCamera(Config *config, Frame *frame, GLFWwindow *window){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "AbstractCamera.cpp\tInitializing\n";
	}
	
	// Init
	this->fov = 70.f;
	this->position = glm::dvec3(0, 0, 0);
	this->keyboard = KeyboardInput::getInstance();
	this->prevX = 0.0;
	this->prevY = 0.0;
	this->prev_initialized = false;
	this->mouse1_pressed = false;
	this->mouse2_pressed = false;
	this->mouse3_pressed = false;
	this->frame = frame;
	this->window = window;
	
	// MVP
	projection = glm::perspectiveFov(float(M_PI)*fov/180.f, 1800.f, 1000.f, 0.001f, 1000000000000.f);
}

AbstractCamera::~AbstractCamera(){
	if((debugLevel & 0x10) == 16){
		std::cout << "AbstractCamera.cpp\tFinalizing\n";
	}
}

void AbstractCamera::checkMouseLocation(void)
{
	if ( !( mouse1_pressed || mouse2_pressed || mouse3_pressed ) )
	{
		/* Only do this checking if a mouse button is pressed */
		return;
	}

	/* Cursor position */
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	/* Window size */
	int width = frame->getWidth();
	int height = frame->getHeight();

	if( xpos < 0.0 || xpos > width )
	{
		/* Setting mouse in center of the window */
		glfwSetCursorPos(window, width/2.0, height/2.0);
		prev_initialized = false;
		return;
	}

	if ( ypos < 0.0 || ypos > height )
	{
		/* Setting mouse in center of the window */
		glfwSetCursorPos(window, width/2.0, height/2.0);
		prev_initialized = false;
	}
}

glm::mat4 AbstractCamera::getVP(void){
	return projection * view;
}

glm::dvec3 AbstractCamera::getPosition(void){
	return position;
}

glm::dvec3 AbstractCamera::getDirection(void){
	return direction;
}

glm::dvec3 AbstractCamera::getUp(void){
	return up;
}

float AbstractCamera::getFieldOfView(void){
	return fov;
}

void AbstractCamera::setFieldOfView(float fov){
	this->fov = fov;
	this->projection = glm::perspectiveFov(float(M_PI)*fov/180.f, 1800.f, 1000.f, 0.001f, 1000000000000.f);
}

void AbstractCamera::setActive(bool active){
	this->active = active;
	
	if(active){
		keyboard->addInputAction(this);
	}else{
		keyboard->removeInputAction(this);
	}
}

void AbstractCamera::onKeyInput(int key){
	(void)key;
}
