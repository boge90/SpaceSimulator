#ifndef BODY_CAMERA_CONTROL_H
#define BODY_CAMERA_CONTROL_H

class BodyCameraControl;

#include "../include/AbstractCamera.hpp"
#include "../include/Frame.hpp"
#include "../include/Config.hpp"
#include "../include/Body.hpp"

class BodyCameraControl: public AbstractCamera{
	private:
		// OpenGL
		GLFWwindow *window;
		Frame *frame;
	
		// Data
		Body *body;
		
		// Misc
		size_t debugLevel;
		
		//Camera data
		double distance;
		double horizontalAngle;
		double verticalAngle;
		double mouseSpeed;
		double previousTime;
	public:
		/**
		* Creates and initializes the body camera
		**/
		BodyCameraControl(GLFWwindow *window, Frame *frame, Body *body, Config *config);
		
		/**
		* Frees up the memory associated with the camera
		**/
		~BodyCameraControl(void);
		
		/**
		* Checks the user input and updates the camera
		**/
		void checkUserInput(void);
		
		/**
		*
		**/
		std::string getCameraName(void);
};

#endif
