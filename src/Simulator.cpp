#include "../include/Simulator.hpp"
#include <iostream>

Simulator::Simulator(double time, std::vector<Body*> *bodies, Config *config){
	// Init
	this->dt = config->getDt();
	this->debugLevel = config->getDebugLevel();
	this->time = time;
	this->bodies = bodies;
	this->simulationSteps = 0;
	this->paused = false;
	
	// Debug
	if((debugLevel & 0x10) == 16){
		std::cout << "Simulator.cpp\t\tInitializing\n";
	}
	
	// Initializing visualization system
	renderer = new Renderer(this, config);
	frame = new Frame(1500, 900, "Space", renderer, this, config);
	
	// Setting bodies for rendering and initializes them
	for(unsigned int i=0; i<bodies->size(); i++){
		(*bodies)[i]->init();
		renderer->addRenderable((*bodies)[i]);
	}
	
	// Adding cameras as renderable
	std::vector<AbstractCamera*> *cameras = frame->getMenu()->getCameras();
	for(size_t i=0; i<cameras->size(); i++){
		renderer->addRenderable((*cameras)[i]);
	}
	
	// Initializing sub renderers
	bodyTracer = new BodyTracer(bodies, config);
	renderer->addRenderable(bodyTracer);
	
	// Initializing sub simulators
	nbody = new Nbody(bodies, config);
	bodyRotator = new BodyRotator(bodies, config);
	rayTracer = new RayTracer(bodies, config);
}

Simulator::~Simulator(){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "Simulator.cpp\t\tFinalizing after executing " << simulationSteps << " simulations steps (" << (simulationSteps*dt)/(3600.0) << " hours)\n";
	}
	
	// Free
	delete frame;
	delete renderer;
	delete nbody;
	delete bodyRotator;
	delete rayTracer;
	delete bodyTracer;
}

void Simulator::simulate(void){	
	if(!paused){	
		// Sub - simulations
		nbody->simulateGravity();
		rayTracer->simulateRays();
		bodyRotator->simulateRotation();

		// Misc
		simulationSteps++;
		time += dt;
	}

	// Check user input IFF menu is hidden
	if(!(frame->getMenu()->isHudVisible())){	
		frame->getMenu()->getActivatedCamera()->checkUserInput();
	}
	
	// Update visualization
	frame->update();
}

double Simulator::getTime(void){
	return time;
}

void Simulator::setPaused(bool paused){
	this->paused = paused;
}

std::vector<Body*>* Simulator::getBodies(void){
	return bodies;
}

bool Simulator::isRunning(){
	return !frame->isClosed();
}

Nbody* Simulator::getNbodySimulator(void){
	return nbody;
}

RayTracer* Simulator::getRayTracerSimulator(void){
	return rayTracer;
}

BodyTracer* Simulator::getBodyTracer(void){
	return bodyTracer;
}

Frame* Simulator::getFrame(void){
	return frame;
}
