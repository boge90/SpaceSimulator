#ifndef ABSTRACT_CAMERA_H
#define ABSTRACT_CAMERA_H

#include <stdlib.h>
#include <iostream>

#include "../include/common.hpp"
#include "../include/Config.hpp"
#include "../include/Shader.hpp"

// Function prototypes
class AbstractCamera{
	protected:
		// Camera
		float fov;
		glm::mat4 projection;
		glm::mat4 view;

		glm::dvec3 position;
		glm::dvec3 direction;
		glm::dvec3 up;
		
		size_t debugLevel;
		bool active;
	public:
		/**
		* Creates the camera and initializes it in vec3(0, 0, 0)
		**/
		AbstractCamera(Config *config);
		
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
