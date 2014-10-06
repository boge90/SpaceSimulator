#ifndef SELECT_VIEW_H
#define SELECT_VIEW_H

template <class T> class SelectView;

#include "../include/common.hpp"
#include "../include/Layout.hpp"
#include "../include/TextView.hpp"
#include "../include/Button.hpp"
#include "../include/Config.hpp"
#include "../include/ViewClickedAction.hpp"
#include "../include/SelectViewStateChangeAction.hpp"

#include <iostream>
#include <string>
#include <vector>

template <class T> class SelectView: public Layout, public ViewClickedAction{
	private:
		// Debug
		size_t debugLevel;
		
		// GUI
		std::string text;
		bool repaint;
		Button *nextButton;
		Button *prevButton;
		TextView *textView;
		TextView *titleView;
		
		// List
		size_t activeElement;
		std::vector<std::string> *names;
		std::vector<T> *values;
		
		// Listeners
		std::vector<SelectViewStateChangeAction<T>*> *listeners;
		
		/**
		*
		**/
		void fireListeners(T t){
			for(size_t i=0; i<listeners->size(); i++){
				(*listeners)[i]->onStateChange(this, t);
			}
		}
	public:
		/**
		*
		**/
		SelectView(std::string text, Config *config): Layout(0, 0, 1, 30, config){
			// Debug
			this->debugLevel = config->getDebugLevel();
			if((debugLevel & 0x10) == 16){
				std::cout << "SelectView.cpp\t\tInitializing" << std::endl;
			}
	
			// Init
			this->text = text;
			this->repaint = false;
			this->activeElement = 0;
			this->names = new std::vector<std::string>();
			this->values = new std::vector<T>();
			this->nextButton = new Button("NEXT", config);
			this->prevButton = new Button("PREV", config);
			this->textView = new TextView("........", config);
			this->titleView = new TextView(text, config);
			this->listeners = new std::vector<SelectViewStateChangeAction<T>*>();
			
			// Listeners
			nextButton->addViewClickedAction(this);
			prevButton->addViewClickedAction(this);
			
			// Adding children such that they can be clicked
			addChild(nextButton);
			addChild(prevButton);
			addChild(titleView);
			addChild(textView);
		}
		
		/**
		*
		**/
		~SelectView(void){
			if((debugLevel & 0x10) == 16){
				std::cout << "SelectView.cpp\t\tFinalizing" << std::endl;
			}
	
			delete names;
			delete values;
			delete listeners;
		}
		
		/**
		*
		**/
		void addItem(std::string name, T t){
			names->push_back(name);
			values->push_back(t);
			
			textView->setText((*names)[activeElement]);
		}
		
		/**
		*
		**/
		void addSelectViewStateChangeAction(SelectViewStateChangeAction<T> *listener){
			listeners->push_back(listener);
		}
		
		/**
		*
		**/
		T* getSelectedItem(void){
			return (*values)[activeElement];
		}
		
		/**
		*
		**/
		void relocate(int x, int y, int width, int height){
			// Super
			View::relocate(x, y, width, height);
			
			x +=5; y+=5;
			titleView->relocate(x, y, DrawService::widthOf(text) + 20, 20);
			
			x += DrawService::widthOf(text) + 20 + 5;
			prevButton->relocate(x, y, 50, 20);
			
			x += 50 + 5;
			nextButton->relocate(x, y, 50, 20);
			
			x += 50 + 5;
			textView->relocate(x, y, 50, 20);
		}
		
		/**
		*
		**/
		void draw(DrawService *drawService){
			// Super
			View::draw(drawService);
			
			int x = this->x + 135 + DrawService::widthOf(text) + 5;
			int y = this->y + 5;
			textView->relocate(x, y, drawService->widthOf((*names)[activeElement]) + 20, 20);
			textView->setText((*names)[activeElement]);
			textView->repaint();
				
			titleView->draw(drawService);
			prevButton->draw(drawService);
			textView->draw(drawService);
			nextButton->draw(drawService);
		}
		
		/**
		*
		**/
		void onClick(View *view, int button, int action){
			if(view == prevButton && activeElement > 0 && action == GLFW_PRESS){
				activeElement--;
				fireListeners((*values)[activeElement]);
			}else if(view == nextButton && activeElement < values->size()-1 && action == GLFW_PRESS){
				activeElement++;
				fireListeners((*values)[activeElement]);
			}
		}
};

#endif
