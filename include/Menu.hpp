#ifndef MENU_H
#define MENU_H

#include <GL/glew.h> // Must be included such that glew.h ALWAYS is included before gl.h
#include <GLFW/glfw3.h>
#include <GL/gl.h>

class Menu;

#include "../include/Renderable.hpp"
#include "../include/BodyCameraControl.hpp"
#include "../include/FreeCameraControl.hpp"
#include "../include/AbstractCamera.hpp"
#include "../include/Simulator.hpp"
#include "../include/Config.hpp"
#include "../include/Hud.hpp"

class Menu{
	private:
		// Camera
		FreeCameraControl *freeCameraControl;
		std::vector<BodyCameraControl*> *bodyCameraControllers;
		AbstractCamera *activeCamera;
		unsigned int currentActive;
		
		// Misc
		size_t debugLevel;
		
		// HUD
		HUD *hud;
	public:
		/**
		*
		**/
		Menu(GLFWwindow *window, Simulator *simulator, Config *config);
		
		/**
		*
		**/
		~Menu(void);
				
		/**
        * Uses OpenGL to draw this body with the current shader
        **/
        void render(void);
        
        /**
        * Returns the camera activated
        **/
        AbstractCamera* getActivatedCamera(void);
        
        /**
        * Changes the active camera type
        **/
        void changeCamera(bool next);
        
        /**
        * Called when the mouse callback function is called
        **/
        void menuClicked(int button, int action, int x, int y);
        
        /**
        * Toggle the visibility of the HUD
        **/
        void toggleHUD(void);
        
        /**
        * Returns the visibility of the HUD
        **/
        bool isHudVisible(void);
};

#endif
