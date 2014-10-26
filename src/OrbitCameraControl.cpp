#include "../include/OrbitCameraControl.hpp"
#include <iostream>

OrbitCameraControl::OrbitCameraControl(GLFWwindow *window, Frame *frame, Body *body, Config *config): AbstractCamera(config){
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "BodyCameraControl.cpp\tInitializing for body " << body << "\n";
	}

	this->body = body;
	this->window = window;
	this->frame = frame;
	
	this->height = 1.0;
	this->latitude = 0.0;
	this->longitude = 0.0;
	this->horizontal = 0.0;
	this->vertical = 0.0;
	this->speed = 500000;
	this->mouseSpeed = 0.05;
	this->previousTime = 0;
	this->dt = config->getDt();
}

OrbitCameraControl::~OrbitCameraControl(void){

}

void OrbitCameraControl::checkUserInput(void){
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

		// Reset mouse position for next frame
		int frameWidth = frame->getWidth();
		int frameHeight = frame->getHeight();
		glfwSetCursorPos(window, frameWidth/2, frameHeight/2);

		// Compute new orientation
		horizontal += mouseSpeed * deltaTime * double(frameWidth/2.0 - xpos);
		vertical += mouseSpeed * deltaTime * double(frameHeight/2.0 - ypos);

		if(vertical > M_PI/2.0){
			vertical = M_PI/2.0;
		}else if(vertical < -M_PI/2.0){
			vertical = -M_PI/2.0;
		}

		// Flip upside down check must be compared to UP vector
	}else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE){
		// Showing mouse cursor again after left mouse button is released
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// Up vector
	up = glm::dvec3(cos(latitude) * sin(longitude), sin(latitude), cos(latitude) * cos(longitude));
	
	// Transformed UP
	glm::dvec4 tempUp = glm::dvec4(up, 0.0);
	tempUp = glm::rotate(glm::dmat4(1.0), body->getInclination(), glm::dvec3(0.0, 0.0, 1.0)) * glm::rotate(glm::dmat4(1.0), body->getRotation(), glm::dvec3(0.0, 1.0, 0.0)) * tempUp;
	up = glm::dvec3(tempUp);
	
	// View direction
	double lon = (M_PI/2.0) - atan(tempUp.z / tempUp.x);
	double lat = (M_PI/2.0) - acos(tempUp.y / glm::length(tempUp));
	if(tempUp.x < 0){lon += M_PI;}
	glm::dvec4 tempDirection = glm::dvec4(cos(vertical) * sin(horizontal), sin(vertical), cos(vertical) * cos(horizontal), 0.0);
	glm::dmat4 m1 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-lat), glm::dvec3(0.0, 0.0, 1.0));
	glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-lon), glm::dvec3(0.0, 1.0, 0.0));

	// Rotating to earth position
	tempDirection = m1*tempDirection;
	tempDirection = m2*tempDirection;
	direction = glm::dvec3(tempDirection);
	
	// Position
	position = body->getCenter() + up*body->getRadius()*height;
	
	speed = body->getRadius()*height - body->getRadius();
	
	// Move forward	
	if(glfwGetKey(window, 'W') == GLFW_PRESS){
		glm::dvec4 tempDirection = glm::dvec4(cos(vertical) * sin(horizontal), sin(vertical), cos(vertical) * cos(horizontal), 0.0);
		glm::dmat4 m1 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-latitude), glm::dvec3(0.0, 0.0, 1.0));
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-longitude), glm::dvec3(0.0, 1.0, 0.0));
		
		tempDirection = m1*tempDirection;
		tempDirection = m2*tempDirection;
		glm::dvec3 dir = glm::dvec3(tempDirection);
		
		// Local pos
		glm::dvec3 tempup = glm::dvec3(cos(latitude) * sin(longitude), sin(latitude), cos(latitude) * cos(longitude));
		
		glm::dvec3 tempPosition = body->getCenter() + tempup*body->getRadius()*height;
		tempPosition += glm::normalize(dir) * speed * deltaTime;
		
		glm::dvec3 temp = tempPosition - body->getCenter();
		
		longitude = (M_PI/2.0) - atan(temp.z / temp.x);
		latitude = (M_PI/2.0) - acos(temp.y / glm::length(temp));
		
		if(temp.x < 0){
			longitude += M_PI;
		}
	}
	// Move backward
	if(glfwGetKey(window, 'S') == GLFW_PRESS){
		glm::dvec4 tempDirection = glm::dvec4(cos(vertical) * sin(horizontal), sin(vertical), cos(vertical) * cos(horizontal), 0.0);
		glm::dmat4 m1 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-latitude), glm::dvec3(0.0, 0.0, 1.0));
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-longitude), glm::dvec3(0.0, 1.0, 0.0));
		
		tempDirection = m1*tempDirection;
		tempDirection = m2*tempDirection;
		glm::dvec3 dir = glm::dvec3(tempDirection);
		
		// Local pos
		glm::dvec3 tempup = glm::dvec3(cos(latitude) * sin(longitude), sin(latitude), cos(latitude) * cos(longitude));
		
		glm::dvec3 tempPosition = body->getCenter() + tempup*body->getRadius()*height;
		tempPosition -= glm::normalize(dir) * speed * deltaTime;
		
		glm::dvec3 temp = tempPosition - body->getCenter();
		
		longitude = (M_PI/2.0) - atan(temp.z / temp.x);
		latitude = (M_PI/2.0) - acos(temp.y / glm::length(temp));
		
		if(temp.x < 0){
			longitude += M_PI;
		}
	}
	// Strafe right
	if(glfwGetKey(window, 'D') == GLFW_PRESS){
		glm::dvec4 tempDirection = glm::dvec4(cos(vertical) * sin(horizontal), sin(vertical), cos(vertical) * cos(horizontal), 0.0);
		glm::dmat4 m1 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-latitude), glm::dvec3(0.0, 0.0, 1.0));
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-longitude), glm::dvec3(0.0, 1.0, 0.0));
		
		tempDirection = m1*tempDirection;
		tempDirection = m2*tempDirection;
		glm::dvec3 dir = glm::dvec3(tempDirection);
		
		// Local pos
		glm::dvec3 tempup = glm::dvec3(cos(latitude) * sin(longitude), sin(latitude), cos(latitude) * cos(longitude));
		glm::dvec3 right = glm::cross(tempup, dir);
		
		glm::dvec3 tempPosition = body->getCenter() + tempup*body->getRadius()*height;
		tempPosition -= glm::normalize(right) * speed * deltaTime;
		
		glm::dvec3 temp = tempPosition - body->getCenter();
		
		longitude = (M_PI/2.0) - atan(temp.z / temp.x);
		latitude = (M_PI/2.0) - acos(temp.y / glm::length(temp));
		
		if(temp.x < 0){
			longitude += M_PI;
		}
	}
	// Strafe left
	if(glfwGetKey(window, 'A') == GLFW_PRESS){
		glm::dvec4 tempDirection = glm::dvec4(cos(vertical) * sin(horizontal), sin(vertical), cos(vertical) * cos(horizontal), 0.0);
		glm::dmat4 m1 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-latitude), glm::dvec3(0.0, 0.0, 1.0));
		glm::dmat4 m2 = glm::rotate(glm::dmat4(1.0), -(M_PI/2.0-longitude), glm::dvec3(0.0, 1.0, 0.0));
		
		tempDirection = m1*tempDirection;
		tempDirection = m2*tempDirection;
		glm::dvec3 dir = glm::dvec3(tempDirection);
		
		// Local pos
		glm::dvec3 tempup = glm::dvec3(cos(latitude) * sin(longitude), sin(latitude), cos(latitude) * cos(longitude));
		glm::dvec3 right = glm::cross(tempup, dir);
		
		glm::dvec3 tempPosition = body->getCenter() + tempup*body->getRadius()*height;
		tempPosition += glm::normalize(right) * speed * deltaTime;
		
		glm::dvec3 temp = tempPosition - body->getCenter();
		
		longitude = (M_PI/2.0) - atan(temp.z / temp.x);
		latitude = (M_PI/2.0) - acos(temp.y / glm::length(temp));
		
		if(temp.x < 0){
			longitude += M_PI;
		}
	}
	// Climb
	if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS){
		height *= 1.001;
	}
	// Go down
	if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
		height /= 1.001;
		if(height < 1.0){height = 1.0;}
	}

	view = glm::lookAt(glm::dvec3(0.0, 0.0, 0.0), direction, up);	
	
	previousTime = currentTime;
}

std::string OrbitCameraControl::getCameraName(void){
	std::string text = *body->getName();
	text.append(" ORBIT CAMERA");
	return text;
}

