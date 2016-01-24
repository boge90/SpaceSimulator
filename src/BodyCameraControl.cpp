#include "../include/BodyCameraControl.hpp"
#include <iostream>

BodyCameraControl::BodyCameraControl(GLFWwindow *window, Frame *frame, Body *body, Config *config): AbstractCamera(config, frame, window){
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "BodyCameraControl.cpp\tInitializing for body " << body << "\n";
	}

	this->body = body;
	this->window = window;
	
	this->flipCheck = config->isFlipCheck();
	this->horizontalAngle = 0.f;
	this->verticalAngle = 0.f;
	this->mouseSpeed = 0.05f * config->getMouseSpeed();
	this->previousTime = 0;
	this->distance = 10.f;
}

BodyCameraControl::~BodyCameraControl(void){
	if((debugLevel & 0x10) == 16){	
		std::cout << "BodyCameraControl.cpp\tFinalizing for body " << body << "\n";
	}
}

void BodyCameraControl::onKeyInput(int key){
	AbstractCamera::onKeyInput(key);
	
	// Delta time
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - previousTime);
	
	if(key == 263){ // LEFT ARROW
		horizontalAngle -= mouseSpeed * deltaTime * 20;
	}else if(key == 262){ // RIGHT ARROW
		horizontalAngle += mouseSpeed * deltaTime * 20;
	}else if(key == 265){ // UP ARROW
		verticalAngle += mouseSpeed * deltaTime * 20;
	}else if(key == 264){ // DOWN ARROW
		verticalAngle -= mouseSpeed * deltaTime * 20;
	}else if(key == 45){ // ZOOM IN
		distance += mouseSpeed * deltaTime * 50;
	}else if(key == 47){ // ZOOM OUT
		distance -= mouseSpeed * deltaTime * 50;
	}
}

void BodyCameraControl::checkUserInput(void){
	// Delta time
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - previousTime);
	
	// Zoom
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS){
		// Hiding cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		// Get mouse position
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		/* Initialize */
		if ( !prev_initialized )
		{
			prevX = xpos;
			prevY = ypos;
			prev_initialized = true;
			return;
		}

		// Changing the distance to the body
		distance -= (mouseSpeed*2) * deltaTime * float(prevY - ypos);
		prevY = ypos;
		mouse3_pressed = true;

	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_RELEASE && mouse3_pressed){
		// Showing mouse cursor again after mouse button 3 is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		prevY = 0.0;
		prevX = 0.0;
		prev_initialized = false;
		mouse3_pressed  = false;
	}
	
	// Click check, will only move camera if left mouse button is pressed
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS){
		// Hiding mouse cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		// Get mouse position
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		/* Initialize */
		if ( !prev_initialized )
		{
			prevX = xpos;
			prevY = ypos;
			prev_initialized = true;
			return;
		}

		// Compute new orientation
		horizontalAngle += mouseSpeed * deltaTime * double(prevX - xpos);
		verticalAngle -= mouseSpeed * deltaTime * double(prevY - ypos);

		prevX = xpos;
		prevY = ypos;
		mouse2_pressed = true;

		// Flip upside down check
		if(flipCheck){		
			if(verticalAngle > M_PI/2.0){
				verticalAngle = M_PI/2.0;
			}else if(verticalAngle < -M_PI/2.0){
				verticalAngle = -M_PI/2.0;
			}
		}
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE && mouse2_pressed){
		// Showing mouse cursor again after left mouse button is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		prevY = 0.0;
		prevX = 0.0;
		prev_initialized = false;
		mouse2_pressed = false;
	}
	   
	// Direction : Spherical coordinates to Cartesian coordinates conversion
	direction = glm::dvec3(cos(verticalAngle) * sin(horizontalAngle), sin(verticalAngle), cos(verticalAngle) * cos(horizontalAngle));
	direction *= body->getRadius()*distance;
	
	// Right vector
	glm::dvec3 right = glm::dvec3(sin(horizontalAngle + M_PI/2.0), 0, cos(horizontalAngle + M_PI/2.0));
	
	glm::dvec3 bodyCenter = body->getCenter();
	position = bodyCenter+direction;
	direction = -direction;
	up = glm::cross(right, direction);
	
	view = glm::lookAt(glm::dvec3(0.0, 0.0, 0.0), direction, up);
	
	previousTime = currentTime;
}

std::string BodyCameraControl::getCameraName(void){
	std::string text = *body->getName();
	text.append(" CAMERA");
	return text;
}
