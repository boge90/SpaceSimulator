#ifndef ABSTRACT_CAMERA_H
#define ABSTRACT_CAMERA_H

#include <stdlib.h>
#include <iostream>

#include "../include/common.hpp"

// Function prototypes
class AbstractCamera{
	protected:
		// Camera
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;

	public:
		/**
		* Creates the camera and initializes it in vec3(0, 0, 0)
		**/
		AbstractCamera();
		
		/**
		* Finalizes the camera
		**/
		virtual ~AbstractCamera(void) = 0;
		
		/**
		* Returns the Model View Projection matrix
		**/
		glm::mat4 getMVP(void);
		
		/**
		* This is called from the Simulator
		**/
		virtual void checkUserInput(void) = 0;
		
		/**
		* Called when the camera is actived
		**/
		virtual void activated(void);
};
#endif
