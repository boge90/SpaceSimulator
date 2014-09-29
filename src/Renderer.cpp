#include "../include/Renderer.hpp"

Renderer::Renderer(Simulator *simulator, Config *config){
	// Init
	this->debugLevel = config->getDebugLevel();	

	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "Renderer.cpp\t\tInitializing\n";
	}
	
	// Initialize
	renderables = new std::vector<Renderable*>();
	this->simulator = simulator;
}

Renderer::~Renderer(){
	if((debugLevel & 0x10) == 16){
		std::cout << "Renderer.cpp\t\tFinalizing\n";
	}
	
	delete renderables;
}

void Renderer::addRenderable(Renderable *renderable){
	if((debugLevel & 0x8) == 8){
		std::cout << "Renderer.cpp\t\tAdding renderable " << renderable << "\n";
	}

	renderables->push_back(renderable);
}

void Renderer::render(glm::mat4 *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up){
	// Drawing renderables
	int size = renderables->size();
	
	for(int i=0; i<size; i++){	
		Renderable *renderable = renderables->at(i);
		renderable->render(mvp, position, direction, up);
	}
}
