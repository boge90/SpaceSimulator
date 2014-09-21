#ifndef RENDERER_H
#define RENDERER_H

class Renderer;

#include "../include/Body.hpp"
#include "../include/Simulator.hpp"
#include "../include/common.hpp"

#include <vector>
#include <iostream>

class Renderer{
	private:
		std::vector<Renderable*> *renderables;
		Simulator *simulator;
		
	public:
		/**
		* Initializes the rendering class
		**/
		Renderer(Simulator *simulator);
		
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
