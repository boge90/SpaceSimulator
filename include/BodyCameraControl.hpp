#ifndef BODY_CAMERA_CONTROL_H
#define BODY_CAMERA_CONTROL_H

class BodyCameraControl;

#include "../include/AbstractCamera.hpp"
#include "../include/Frame.hpp"
#include "../include/Body.hpp"

class BodyCameraControl: public AbstractCamera{
	private:
		// OpenGL
		GLFWwindow *window;
		Frame *frame;
	
		// Data
		Body *body;
		
		//Camera data
		float distance;
		float horizontalAngle;
		float verticalAngle;
		float mouseSpeed;
		double previousTime;
	public:
		/**
		* Creates and initializes the body camera
		**/
		BodyCameraControl(GLFWwindow *window, Frame *frame, Body *body);
		
		/**
		* Frees up the memory associated with the camera
		**/
		~BodyCameraControl(void);
		
		/**
		* Checks the user input and updates the camera
		**/
		void checkUserInput(void);
};

#endif
