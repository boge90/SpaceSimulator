#include "../include/ListLayout.hpp"
#include <iostream>

ListLayout::ListLayout(int x, int y, int width, int height, Config *config): ScrollLayout(x, y, width, height, config)
{
	// Debug
	if((debugLevel & 0x10) == 16)
	{
		std::cout << "ListLayout.cpp\t\tInitializing" << std::endl;
	}
}

ListLayout::~ListLayout()
{
	// Debug
	if((debugLevel & 0x10) == 16)
	{
		std::cout << "ListLayout.cpp\t\tFinalizing" << std::endl;
	}
}

void ListLayout::addChild(View *view)
{	
	//Super
	Layout::addChild(view);

	int padding = 10;
	size_t childNumber = children->size();
	
	// Find Y coordinate
	int childY = y + padding; // Top padding
	for(size_t i=0; i<childNumber-1; i++)
	{
		childY += (*children)[i]->getHeight() + padding; // 10 is padding
	}
	
	view->relocate(x + padding, childY, width - 2*padding, 30);
}

void ListLayout::scroll( double xoffset, double yoffset )
{
	/* Super */
	ScrollLayout::scroll( xoffset, yoffset );

	/* Move self */
	scrolled_y += yoffset*10;

	/* Move children */
	size_t num_children = children->size();
	for( size_t i=0; i<num_children; i++ )
	{
		int child_x = (*children)[i]->getX();
		int child_y = (*children)[i]->getY();
		int child_width = (*children)[i]->getWidth();
		int child_height = (*children)[i]->getHeight();

		child_y += (int)yoffset*10;

		(*children)[i]->relocate( child_x, child_y, child_width, child_height );
	}
}
