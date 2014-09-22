#ifndef INPUT_VIEW_H
#define INPUT_VIEW_H

class InputView;

#include "../include/Button.hpp"
#include "../include/InputViewAction.hpp"
#include "../include/ViewClickedAction.hpp"
#include "../include/KeyboardInputAction.hpp"

#include <string>
#include <vector>

class InputView: public Button, public ViewClickedAction, public KeyboardInputAction{
	private:
		bool repaint;
		std::vector<InputViewAction*> *listeners;

	protected:
		std::string input;
		
	public:
		/**
		* Creates the input view object
		**/
		InputView(std::string text);
		
		/**
		* Finalizes the input view object
		**/
		~InputView(void);
		
		/**
		* Called when the user clicks the field
		**/
		void onClick(View *view, int button, int action);
		
		/**
		* Called when the user has activated the field, and
		* pushes keyboard buttons
		**/
		void onKeyInput(int key);
		
		/**
		* Overriders the parent class draw function such that the input text is drawn
		**/
		void draw(DrawService *drawService);
		
		/**
		* Adds a listener for the user input
		**/
		void addInputViewAction(InputViewAction *listener);
		
		/**
		* sets the field input value
		**/
		void setInput(std::string input);
};

#endif
