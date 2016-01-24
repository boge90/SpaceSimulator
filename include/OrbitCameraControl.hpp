#ifndef ORBIT_CAMERA_H
#define ORBIT_CAMERA_H

#include "../include/Body.hpp"
#include "../include/Frame.hpp"
#include "../include/AbstractCamera.hpp"

class OrbitCameraControl: public AbstractCamera{
	private:
		// Debug
		size_t debugLevel;
	
		Body *body;
		GLFWwindow *window;
		
		// Camera data
		double longitude, latitude; // Position on body
		double vertical, horizontal; // Camera view direction
		double speed, mouseSpeed;
		double previousTime;
		double height;
		double *dt;
	public:
		/**
		* Constructs this camera
		**/
		OrbitCameraControl(GLFWwindow *window, Frame *frame, Body *body, Config *config);
		
		/**
		* Delete this camera
		**/
		~OrbitCameraControl();
		
		/**
		* Check the user input
		**/
		void checkUserInput(void);
		
		/**
		* Returns the name of this camera (Used in HUD)
		**/
		std::string getCameraName(void);
};

#endif
