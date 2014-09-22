#ifndef INPUT_VIEW_ACTION_H
#define INPUT_VIEW_ACTION_H

#include "../include/InputView.hpp"

class InputViewAction{
	public:
		virtual void onInput(InputView *view, std::string *input) = 0;
};

#endif
