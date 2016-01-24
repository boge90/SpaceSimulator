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
	this->renderer = new Renderer(this, config);
	this->frame = new Frame(1800, 1000, "Space", renderer, this, config);
	this->keyboard = new Keyboard(bodies, config);
	
	// Initializing sub renderers
	this->skybox = new Skybox(config);
	this->bodyTracer = new BodyTracer(bodies, config);
	this->bodyLocator = new BodyLocator(bodies, config);
	this->bodyLevelOfDetail = new BodyLevelOfDetail(bodies, config);
	
	// Adding renderables
	this->renderer->addRenderable(skybox);
	this->renderer->addRenderable(bodyTracer);
	for(unsigned int i=0; i<bodies->size(); i++){
		(*bodies)[i]->init();
		this->renderer->addRenderable((*bodies)[i]);
	}
	this->renderer->addRenderable(bodyLocator);
	
	// Initializing sub simulators
	this->nbody = new Nbody(bodies, config);
	this->bodyRotator = new BodyRotator(bodies, config);
	this->rayTracer = new RayTracer(bodies, config);
	this->starDimmer = new StarDimmer(this, bodies, config);

	// Misc checkers
	this->deltaTimeChecker = new DeltaTimeChecker( config );
}

Simulator::~Simulator(){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "Simulator.cpp\t\tFinalizing" << std::endl;
	}
	
	std::cout << "Simulator.cpp\t\tExecuted " << simulationSteps << " simulations steps (" << (simulationSteps* (*dt) )/(3600.0) << " hours)\n";
	
	// Free
	delete keyboard;
	delete frame;
	delete renderer;
	delete nbody;
	delete bodyRotator;
	delete rayTracer;
	delete bodyTracer;
	delete skybox;
	delete bodyLocator;
	delete starDimmer;
	delete bodyLevelOfDetail;
	delete deltaTimeChecker;
}

void Simulator::simulate(void){
	if(!paused){
		// Sub - simulations
		nbody->simulateGravity();
		bodyRotator->simulateRotation();
		rayTracer->simulateRays();
	}
	
	// Check user input IFF menu is hidden
	if(!(frame->getHud()->isVisible())){
		frame->getHud()->getActivatedCamera()->checkUserInput();
		frame->getHud()->getActivatedCamera()->checkMouseLocation();
	}
	
	// Simulations that need updated camera position
	if(!paused){
		glm::dvec3 position = frame->getHud()->getActivatedCamera()->getPosition();
		starDimmer->simulateStarDimming(position);
		bodyLevelOfDetail->update(position);
		
		// Misc
		simulationSteps++;
		time += *dt;
	}
	
	// Update visualization
	frame->update();

	// Misc checkers
	double fps = frame->getFPS();
	deltaTimeChecker->update_delta_time_if_needed( fps, time );
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

BodyLocator* Simulator::getBodyLocator(void){
	return bodyLocator;
}

Frame* Simulator::getFrame(void){
	return frame;
}

Skybox* Simulator::getSkybox(void){
	return skybox;
}

StarDimmer* Simulator::getStarDimmer(void){
	return starDimmer;
}
