#ifndef FRAME_H
#define FRAME_H

// Includes
#include <stdio.h>
#include <GL/glew.h> // Must be included such that glew.h ALWAYS is included before gl.h
#include <GLFW/glfw3.h>
#include <GL/gl.h>

class Frame;

#include "../include/AbstractCamera.hpp"
#include "../include/Shader.hpp"
#include "../include/Renderer.hpp"
#include "../include/Simulator.hpp"
#include "../include/Config.hpp"
#include "../include/KeyboardInput.hpp"
#include "../include/Hud.hpp"

class Frame{
	private:
		// GLFW
		GLFWwindow *window;
		HUD *hud;
		
		// Simulator
		Simulator *simulator;

		// 3D
		Renderer *renderer;
		long unsigned int totalFrameCount;
		long unsigned int lastSecondFrameCount;
		double prevTime;
		double fps;
		
		// Misc
		size_t debugLevel;
		GLuint VertexArrayID;
		static int frameWidth;
		static int frameHeight;
		static Frame *instance;
		KeyboardInput *keyboardInput;
		
		/**
		* Callback function when clicking the close button on the frame
		**/
		static void windowCloseCallback(GLFWwindow *window);
		
		/**
		* Callback function for when the window size is changed
		**/
		static void windowSizeChangeCallback(GLFWwindow *window, int width, int height);
		
		/**
		* Callback function for keyboard input
		**/
		static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
		
		/**
		*
		**/
		static void mouseCallback(GLFWwindow *window, int button, int action, int mods);
	public:
		/**
		* Constructor
		**/
		Frame(int width, int height, const char *title, Renderer *renderer, Simulator *simulator, Config *config);
		
		/**
		* Finalizes the frame
		**/
		~Frame();
		
		/**
		* If the frame is visible, this function unlocks the updateMutex, such that the GUI thread can update the frame
		**/
		void update(void);
		
		/**
		* Returns the closed flag
		**/
		bool isClosed(void);
		
		/**
		* Returns the frame count (number of times that 'update' has been called)
		**/
		long unsigned int getFrameCount(void);
		
		/**
		* Returns the current FPS
		**/
		double getFPS(void);
		
		/**
		* Returns the pointer to the Menu
		**/
		HUD* getHud();
		
		/**
		* Returns the pointer to the simulator
		**/
		Simulator* getSimulator(void);
		
		/**
		* Returns the current width of the frame
		**/
		int getWidth(void);
		
		/**
		* Returns the current height of the frame
		**/
		int getHeight(void);
		
		/**
		* Returns the pointer to the GLFW window object
		**/
		GLFWwindow* getWindow(void);
};
#endif

