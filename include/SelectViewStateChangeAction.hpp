#ifndef SELECT_VIEW_STATE_CHANGE_ACTION
#define SELECT_VIEW_STATE_CHANGE_ACTION

template <class T> class SelectViewStateChangeAction;

#include "../include/SelectView.hpp"

template <class T> class SelectViewStateChangeAction{
	public:
		virtual void onStateChange(SelectView<T> *view, T t) = 0;
};

#endif
