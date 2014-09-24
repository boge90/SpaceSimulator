#ifndef FLOAT_VIEW_INPUT_H
#define FLOAT_VIEW_INPUT_H

class FloatInputView;

#include "../include/Config.hpp"
#include "../include/InputView.hpp"
#include "../include/InputViewAction.hpp"
#include "../include/FloatInputAction.hpp"

#include <vector>

class FloatInputView: public InputView, public InputViewAction{
	private:
		std::vector<FloatInputAction*> *listeners;
	public:
		/**
		*
		**/
		FloatInputView(std::string text, Config *config);
		
		/**
		*
		**/
		~FloatInputView();
		
		/**
		*
		**/
		void onInput(InputView *view, std::string *input);
		
		/**
		*
		**/
		void addFloatInputAction(FloatInputAction *listener);
};

#endif
