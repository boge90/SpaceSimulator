#include "../include/BodyCameraControl.hpp"
#include <iostream>

BodyCameraControl::BodyCameraControl(GLFWwindow *window, Frame *frame, Body *body, Config *config): AbstractCamera(config){
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "BodyCameraControl.cpp\tInitializing for body " << body << "\n";
	}

	this->body = body;
	this->window = window;
	this->frame = frame;
	
	this->horizontalAngle = 0.f;
	this->verticalAngle = 0.f;
	this->mouseSpeed = 0.05f;
	this->previousTime = 0;
	this->distance = 10.f;
}

BodyCameraControl::~BodyCameraControl(void){
	if((debugLevel & 0x10) == 16){	
		std::cout << "BodyCameraControl.cpp\tFinalizing for body " << body << "\n";
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

		// Reset mouse position for next frame
		int frameWidth = frame->getWidth();
		int frameHeight = frame->getHeight();
		glfwSetCursorPos(window, frameWidth/2, frameHeight/2);
		
		// Changing the distance to the body
		distance -= (mouseSpeed*2) * deltaTime * float(frameHeight/2.f - ypos);
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_RELEASE){
		// Showing mouse cursor again after mouse button 3 is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
	
	// Click check, will only move camera if left mouse button is pressed
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS){
		// Hiding mouse cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		// Get mouse position
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		// Reset mouse position for next frame
		int frameWidth = frame->getWidth();
		int frameHeight = frame->getHeight();
		glfwSetCursorPos(window, frameWidth/2, frameHeight/2);

		// Compute new orientation
		horizontalAngle += mouseSpeed * deltaTime * double(frameWidth/2.0 - xpos);
		verticalAngle -= mouseSpeed * deltaTime * double(frameHeight/2.0 - ypos);

		// Flip upside down check
		if(verticalAngle > M_PI/2.0){
			verticalAngle = M_PI/2.0;
		}else if(verticalAngle < -M_PI/2.0){
			verticalAngle = -M_PI/2.0;
		}
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE){
		// Showing mouse cursor again after left mouse button is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
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
