#ifndef RENDERER_H
#define RENDERER_H

class Renderer;

#include "../include/Body.hpp"
#include "../include/Simulator.hpp"
#include "../include/common.hpp"
#include "../include/Config.hpp"

#include <vector>
#include <iostream>

class Renderer{
	private:
		// Misc
		size_t debugLevel;
	
		// Renderables
		std::vector<Renderable*> *renderables;
		
		// Simulator
		Simulator *simulator;
		
	public:
		/**
		* Initializes the rendering class
		**/
		Renderer(Simulator *simulator, Config *config);
		
		/**
		* Finalizes the rendering class
		**/
		~Renderer();
		
		/**
		* Sets the bodies that should be rendered
		**/
		void addRenderable(Renderable *renderable);
		
		/**
		* Responsible for calling subsequent render functions for all
		* object that should be rendered
		**/
		void render(const GLfloat *mvp);
};

#endif
