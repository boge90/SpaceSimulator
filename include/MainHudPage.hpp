#ifndef MAIN_HUD_PAGE_H
#define MAIN_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Button.hpp"
#include "../include/Simulator.hpp"

class MainHudPage: public HudPage, public ViewClickedAction{
	private:
		// GUI elements
		Button *wireframeButton;
		Button *futureBodyPathButton;
		Button *exitButton;
		
		// Pointer to the simulator
		Simulator *simulator;
	public:
		/**
		* Creates the main hud page object
		**/
		MainHudPage(int x, int y, int width, int height, Simulator *simulator);
		
		/**
		* finalizes the main HUD page object
		**/
		~MainHudPage();
		
		/**
		* View clicked action listener
		**/
		void viewClicked(View *view, int button, int action);
};

#endif
