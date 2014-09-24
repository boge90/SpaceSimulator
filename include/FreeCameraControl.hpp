#ifndef FREE_CAMERA_CONTROL_H
#define FREE_CAMERA_CONTROL_H

#include "../include/common.hpp"
#include "../include/Config.hpp"
#include "../include/AbstractCamera.hpp"

class FreeCameraControl: public AbstractCamera{
	private:
		//Camera data
		glm::dvec3 position;
		double horizontalAngle;
		double verticalAngle;
		double speed;
		double mouseSpeed;
		double previousTime;
		
		// Window data
		int frameWidth;
		int frameHeight;
		
		// Misc
		size_t debugLevel;
		
		//GLFW
		GLFWwindow *window;
	public:
		/**
		* Constructs the camera object
		**/
		FreeCameraControl(GLFWwindow *window, int frameWidth, int frameHeight, Config *config);
		
		/**
		* Finalizes the camera object
		**/
		~FreeCameraControl();
		
		/**
		* Called from Simulator
		**/
		void checkUserInput(void);
		
		/**
		* Is called when the camera is activated in the menu
		**/
		void activated(void);
};

#endif
