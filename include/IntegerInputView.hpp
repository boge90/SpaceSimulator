#ifndef INTEGER_VIEW_INPUT_H
#define INTEGER_VIEW_INPUT_H

class IntegerInputView;

#include "../include/InputView.hpp"
#include "../include/InputViewAction.hpp"
#include "../include/IntegerInputAction.hpp"

#include <vector>

class IntegerInputView: public InputView, public InputViewAction{
	private:
		std::vector<IntegerInputAction*> *listeners;
	public:
		/**
		*
		**/
		IntegerInputView(std::string text);
		
		/**
		*
		**/
		~IntegerInputView();
		
		/**
		*
		**/
		void onInput(InputView *view, std::string *input);
		
		/**
		*
		**/
		void addIntegerInputAction(IntegerInputAction *listener);
};

#endif
