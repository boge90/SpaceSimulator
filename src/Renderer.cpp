#include "../include/Renderer.hpp"

Renderer::Renderer(Simulator *simulator){
	std::cout << "Renderer.cpp\t\tInitializing\n";
	
	// Initialize
	renderables = new std::vector<Renderable*>();
	this->simulator = simulator;
}

Renderer::~Renderer(){
	std::cout << "Renderer.cpp\t\tFinalizing\n";
	
	delete renderables;
}

void Renderer::addRenderable(Renderable *renderable){
	std::cout << "Renderer.cpp\t\tAdding renderable " << renderable << "\n";

	renderables->push_back(renderable);
}

void Renderer::render(const GLfloat *mvp){
	// Drawing bodies
	int size = renderables->size();
	
	for(int i=0; i<size; i++){	
		Renderable *renderable = renderables->at(i);
		renderable->render(mvp);
	}
}
