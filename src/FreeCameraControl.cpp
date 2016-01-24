#include "../include/FreeCameraControl.hpp"
#include <iostream>

FreeCameraControl::FreeCameraControl(GLFWwindow *window, Frame *frame, Config *config): AbstractCamera(config, frame, window){
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "FreeCameraControl.cpp\tInitializing\n";
	}

	// Initial frame values
	this->window = window;
	this->previousTime = 0;
	this->horizontalAngle = 0;
	this->verticalAngle = 0;
	this->speed = 30000;
	this->mouseSpeed = 0.1 * config->getMouseSpeed();
}

FreeCameraControl::~FreeCameraControl(void){
	if((debugLevel & 0x10) == 16){		
		std::cout << "FreeCameraControl.cpp\tFinalizing\n";
	}
}

void FreeCameraControl::onKeyInput(int key){
	AbstractCamera::onKeyInput(key);
	
	// Delta time
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - previousTime);
	
	if(key == 263){ // LEFT ARROW
		horizontalAngle += mouseSpeed * deltaTime * 20;
	}else if(key == 262){ // RIGHT ARROW
		horizontalAngle -= mouseSpeed * deltaTime * 20;
	}else if(key == 265){ // UP ARROW
		verticalAngle += mouseSpeed * deltaTime * 20;
	}else if(key == 264){ // DOWN ARROW
		verticalAngle -= mouseSpeed * deltaTime * 20;
	}
}

void FreeCameraControl::setActive(bool active){
	// Calling super
	AbstractCamera::setActive(active);
	
	// Debug
	if(active){		
		// preventing glitch bug
		previousTime = glfwGetTime();
	}
}

void FreeCameraControl::checkUserInput(void){
	// Delta time
	double currentTime = glfwGetTime();
	double deltaTime = currentTime - previousTime;
	   
	// Click check, will only move camera if left mouse button is pressed
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS){
		// Hiding mouse cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		// Get mouse position
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		if ( !prev_initialized )
		{
			prevX = xpos;
			prevY = ypos;
			prev_initialized = true;
			return;
		}

		// Compute new orientation
		horizontalAngle += mouseSpeed * deltaTime * (prevX - xpos);
		verticalAngle += mouseSpeed * deltaTime * (prevY - ypos);

		prevX = xpos;
		prevY = ypos;
		mouse1_pressed = true;

		// Flip upside down check
		if(verticalAngle > M_PI/2.0){
			verticalAngle = M_PI/2.0;
		}else if(verticalAngle < -M_PI/2.0){
			verticalAngle = -M_PI/2.0;
		}
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE && mouse1_pressed){
		// Showing mouse cursor again after left mouse button is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		prevX = 0.0;
		prevY = 0.0;
		mouse1_pressed = false;
		prev_initialized = false;
	}
	   
	// Direction : Spherical coordinates to Cartesian coordinates conversion
	direction = glm::dvec3(cos(verticalAngle) * sin(horizontalAngle), sin(verticalAngle), cos(verticalAngle) * cos(horizontalAngle));
	   
	// Right vector
	glm::dvec3 right = glm::dvec3(sin(horizontalAngle - M_PI/2.0), 0, cos(horizontalAngle - M_PI/2.0));
	   
	// Up vector : perpendicular to both direction and right
	up = glm::cross(right, direction);
	
	
	// Increasing speed
	if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
		speed = speed*1.05 + 1.0;
	}
	if(glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS && speed - 1.0 >= 0.0){
		speed /= 1.05;
	}
	   
	// Move forward
	if(glfwGetKey(window, 'W') == GLFW_PRESS){
		position += direction * deltaTime * speed;
	}
	// Move backward
	if(glfwGetKey(window, 'S') == GLFW_PRESS){
		position -= direction * deltaTime * speed;
	}
	// Strafe right
	if(glfwGetKey(window, 'D') == GLFW_PRESS){
		position += right * deltaTime * speed;
	}
	// Strafe left
	if(glfwGetKey(window, 'A') == GLFW_PRESS){
		position -= right * deltaTime * speed;
	}
	// Jump up
	if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS){
		glm::dvec3 temp = glm::dvec3(0.0, 1.0, 0.0);
		position += temp * deltaTime * speed;
	}
	// Go Down
	if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
		glm::dvec3 temp = glm::dvec3(0.0, 1.0, 0.0);
		position -= temp * deltaTime * speed;
	}
	
	view = glm::lookAt(glm::dvec3(0.0, 0.0, 0.0), direction, up);
	
	previousTime = currentTime;
}

std::string FreeCameraControl::getCameraName(void){
	return "FREE CAMERA";
}
