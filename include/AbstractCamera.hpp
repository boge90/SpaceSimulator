#ifndef ABSTRACT_CAMERA_H
#define ABSTRACT_CAMERA_H

#include <stdlib.h>
#include <iostream>

class AbstractCamera;

#include "../include/common.hpp"
#include "../include/Config.hpp"
#include "../include/Shader.hpp"
#include "../include/Frame.hpp"
#include "../include/KeyboardInput.hpp"
#include "../include/KeyboardInputAction.hpp"

// Function prototypes
class AbstractCamera: public KeyboardInputAction{
	protected:
		// Camera
		float fov;
		glm::mat4 projection;
		glm::mat4 view;

		glm::dvec3 position;
		glm::dvec3 direction;
		glm::dvec3 up;

		double prevX;
		double prevY;
		bool prev_initialized;
		bool mouse1_pressed;
		bool mouse2_pressed;
		bool mouse3_pressed;
		
		size_t debugLevel;
		bool active;
		
		// Keyboard
		KeyboardInput *keyboard;

		/* Frame */
		Frame *frame;
		GLFWwindow *window;
		
		/**
		* Called when the user has activated the field, and
		* pushes keyboard buttons
		**/
		virtual void onKeyInput(int key);
	public:
		/**
		* Creates the camera and initializes it in vec3(0, 0, 0)
		**/
		AbstractCamera(Config *config, Frame *frame, GLFWwindow *window);
		
		/**
		* Finalizes the camera
		**/
		virtual ~AbstractCamera(void) = 0;
		
		/**
		* Returns the View Projection matrix
		**/
		glm::mat4 getVP(void);
		
		/**
		* This is called from the Simulator
		**/
		virtual void checkUserInput(void) = 0;
		
		/**
		* Called when the camera is actived
		**/
		virtual void setActive(bool active);
		
		/**
		*
		**/
		void checkMouseLocation(void);
		
		/**
		*
		**/
		glm::dvec3 getPosition(void);
		
		/**
		*
		**/
		glm::dvec3 getDirection(void);
		
		/**
		*
		**/
		glm::dvec3 getUp(void);
		
		/**
		*
		**/
		float getFieldOfView(void);
		
		/**
		*
		**/
		void setFieldOfView(float fov);
		
		/**
		*
		**/
		virtual std::string getCameraName(void) = 0;
};
#endif
