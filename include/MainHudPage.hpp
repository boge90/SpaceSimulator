#ifndef MAIN_HUD_PAGE_H
#define MAIN_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Button.hpp"
#include "../include/CheckBox.hpp"
#include "../include/InputView.hpp"
#include "../include/IntegerInputView.hpp"
#include "../include/CheckBoxStateChangeAction.hpp"
#include "../include/IntegerInputView.hpp"
#include "../include/IntegerInputAction.hpp"
#include "../include/Simulator.hpp"

class MainHudPage: public HudPage, public ViewClickedAction, public CheckBoxStateChangeAction, public IntegerInputAction{
	private:
		// GUI elements
		TextView *fpsView;
		TextView *timeView;
		CheckBox *cullBackfaceBox;
		CheckBox *wireframeBox;
		CheckBox *futureBodyPathBox;
		Button *exitButton;
		IntegerInputView *futureBodyInputView;
		
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
		void onClick(View *view, int button, int action);
		
		/**
		* CheckBox state change listener
		**/
		void onStateChange(CheckBox *box, bool newState);
		
		/**
		* Called when the associated IntergerInputView retrieves input
		**/
		void onIntegerInput(IntegerInputView *view, int value);
		
		/**
		*
		**/
		void draw(DrawService *drawService);
};

#endif
