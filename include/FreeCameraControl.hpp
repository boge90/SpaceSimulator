#ifndef FREE_CAMERA_CONTROL_H
#define FREE_CAMERA_CONTROL_H

#include "../include/common.hpp"
#include "../include/Frame.hpp"
#include "../include/Config.hpp"
#include "../include/AbstractCamera.hpp"

class FreeCameraControl: public AbstractCamera{
	private:
		//Camera data
		double horizontalAngle;
		double verticalAngle;
		double speed;
		double mouseSpeed;
		double previousTime;
		
		// Misc
		size_t debugLevel;
		
		//GLFW
		GLFWwindow *window;
		
		/**
		* Called when this camera is active and the user presses the keyboard
		**/
		void onKeyInput(int key);
	public:
		/**
		* Constructs the camera object
		**/
		FreeCameraControl(GLFWwindow *window, Frame *frame, Config *config);
		
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
		void setActive(bool active);
		
		/**
		*
		**/
		std::string getCameraName(void);
};

#endif
