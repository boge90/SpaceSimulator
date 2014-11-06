#ifndef MAIN_HUD_PAGE_H
#define MAIN_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Button.hpp"
#include "../include/CheckBox.hpp"
#include "../include/InputView.hpp"
#include "../include/IntegerInputView.hpp"
#include "../include/CheckBoxStateChangeAction.hpp"
#include "../include/SelectViewStateChangeAction.hpp"
#include "../include/IntegerInputView.hpp"
#include "../include/SelectView.hpp"
#include "../include/IntegerInputAction.hpp"
#include "../include/Simulator.hpp"
#include "../include/Config.hpp"

class MainHudPage: public HudPage, public ViewClickedAction, public CheckBoxStateChangeAction, public IntegerInputAction, public SelectViewStateChangeAction<int>{
	private:
		// GUI elements
		TextView *fpsView;
		TextView *timeView;
		CheckBox *pausedBox;
		CheckBox *cullBackfaceBox;
		CheckBox *wireframeBox;
		CheckBox *starDimmerBox;
		CheckBox *futureBodyPathBox;
		IntegerInputView *futureBodyInputView;
		CheckBox *bodyLocatorBox;
		IntegerInputView *bodyLocatorInputView;
		SelectView<int> *rayTracingLevelView;
		Button *exitButton;
		
		// Pointer to the simulator
		Simulator *simulator;
	public:
		/**
		* Creates the main hud page object
		**/
		MainHudPage(int x, int y, int width, int height, Simulator *simulator, Config *config);
		
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
		*
		**/
		void onStateChange(SelectView<int> *view, int i);
		
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
