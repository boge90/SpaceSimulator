#include "../include/ScrollLayout.hpp"
#include <iostream>

ScrollLayout::ScrollLayout(int x, int y, int width, int height, Config *config): Layout(x, y, width, height, config)
{
	// Debug
	if((debugLevel & 0x10) == 16)
	{
		std::cout << "ScrollLayout.cpp\t\tInitializing" << std::endl;
	}

	/* Init */
	scrolled_x = 0.0;
	scrolled_y = 0.0;
	has_been_scrolled = false;
}

ScrollLayout::~ScrollLayout()
{
	// Debug
	if((debugLevel & 0x10) == 16)
	{
		std::cout << "ScrollLayout.cpp\t\tFinalizing" << std::endl;
	}
}

void ScrollLayout::scroll( double xoffset, double yoffset )
{
	has_been_scrolled = true;
}

void ScrollLayout::draw(DrawService *drawService){
	// Super
	Layout::draw(drawService);

	if( has_been_scrolled )
	{
		drawService->fill(0, 0, 0);
		has_been_scrolled = false;
	}
}
