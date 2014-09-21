#include "../include/BodyCameraControl.hpp"
#include <iostream>

BodyCameraControl::BodyCameraControl(GLFWwindow *window, Frame *frame, Body *body){
	std::cout << "BodyCameraControl.cpp\tInitializing for body " << body << "\n";

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
	std::cout << "BodyCameraControl.cpp\tFinalizing for body " << body << "\n";
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
		horizontalAngle += mouseSpeed * deltaTime * float(frameWidth/2.f - xpos);
		verticalAngle -= mouseSpeed * deltaTime * float(frameHeight/2.f - ypos);

		// Flip upside down check
		if(verticalAngle > M_PI/2.f){
			verticalAngle = M_PI/2.f;
		}else if(verticalAngle < -M_PI/2.f){
			verticalAngle = -M_PI/2.f;
		}
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE){
		// Showing mouse cursor again after left mouse button is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
	   
	// Direction : Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(cos(verticalAngle) * sin(horizontalAngle), sin(verticalAngle), cos(verticalAngle) * cos(horizontalAngle));
	direction *= body->getRadius()*distance;
	
	// Right vector
	glm::vec3 right = glm::vec3(sin(horizontalAngle - M_PI/2.0f), 0, cos(horizontalAngle - M_PI/2.0f));
	   
	// Up vector : perpendicular to both direction and right
	glm::vec3 up = glm::cross(right, direction);
	
	glm::vec3 bodyCenter = glm::vec3(body->getCenter());
	view = glm::lookAt(bodyCenter+direction, bodyCenter, up);
	
	previousTime = currentTime;
}
