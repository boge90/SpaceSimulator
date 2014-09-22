#ifndef HUD_H
#define HUD_H

class HUD;

#include "../include/common.hpp"
#include "../include/Renderable.hpp"
#include "../include/Shader.hpp"
#include "../include/DrawService.hpp"
#include "../include/HudPage.hpp"
#include "../include/View.hpp"
#include "../include/Simulator.hpp"

#include <vector>

class HUD{
	private:
		// GUI
		View *left;
		View *right;
	
		// Pages
		int activePage;
		std::vector<HudPage*> *pages;
	
		// Texture
		GLuint tex;
		int width;
		int height;
		int stride;
		unsigned char *pixels;
		
		// Helper class for drawing
		DrawService *drawService;
		
		// Visualization
		bool visible;
		GLuint vertexBuffer;
		GLuint texCordsBuffer;
		
		// Shader
		Shader *shader;
		
		// OpenGL
		GLFWwindow *window;
		
		/**
		* Called when finding a layout class in the view tree, for traversing down the tree
		**/
		void hudClickedRecur(int button, int action, int x, int y, Layout *layout);
	public:
		/**
		* Creates the HUD object
		**/
		HUD(GLFWwindow *window, Simulator *simulator);
		
		/**
		* Destroys the HUD object
		**/
		~HUD();
		
		/**
		* Calls the update function for all views, which may have their data modified
		**/
		void update(void);
		
		/**
		* Renders the HUD "framebuffer" on the screen IF visibility flag is true
		**/
		void render(void);
		
		/**
		* Toogles the visibility flag
		**/
		void toggleVisibility(void);
		
		/**
		* Called when the mouse is clicked
		**/
		void hudClicked(int button, int action, int x, int y);
};

#endif
