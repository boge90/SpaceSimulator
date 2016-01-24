#include "../include/CombinedBodyHudPage.hpp"

CombinedBodyHudPage::CombinedBodyHudPage(int x, int y, int width, int height, std::string title, std::vector<Body*> *bodies, Config *config): HudPage(x, y, width, height, title, config)
{
	// Debug
	if((debugLevel & 0x10) == 16){		
		std::cout << "CombinedBodyHudPage.cpp\tInitializing for " << title << std::endl;
	}

	/* Data */
	this->bodies = bodies;

	/* Initializing vectors */
	positionViews = new std::vector<TextView*>();
	velocityViews = new std::vector<TextView*>();
	wireframeBoxes = new std::vector<CheckBox*>();
	visualizationTypeViews = new std::vector<SelectView<Visualization>*>();

	/* Adding GUI elements for each body */
	for( size_t i=0; i<bodies->size(); i++)
	{
		/* Wireframe label */
		std::string checkboxLabel = *((*bodies)[i]->getName());
		std::string visualizationLabel = *((*bodies)[i]->getName());
		checkboxLabel.append(" WIREFRAME");
		visualizationLabel.append(" VISUALIZATION TYPE");

		/* labels */
		TextView *positionView = new TextView("", config);
		TextView *velocityView = new TextView("", config);
		CheckBox *wireframeBox = new CheckBox(checkboxLabel, false, config);
		SelectView<Visualization> *visualization = new SelectView<Visualization>(visualizationLabel, config);
	
		// Listeners
		wireframeBox->addStateChangeAction(this);
		visualization->addSelectViewStateChangeAction(this);

		/* Storing pointers for updating them while drawing */
		positionViews->push_back( positionView );
		velocityViews->push_back( velocityView );
		wireframeBoxes->push_back( wireframeBox );
		visualizationTypeViews->push_back( visualization );

		// Adding visualization types
		visualization->addItem("NORMAL", NORMAL);
		visualization->addItem("TEMPERATURE", TEMPERATURE);

		addChild(positionView);
		addChild(velocityView);
		addChild(wireframeBox);
		addChild(visualization);
	}
}

CombinedBodyHudPage::~CombinedBodyHudPage()
{
	delete positionViews;
	delete velocityViews;
	delete wireframeBoxes;
	delete visualizationTypeViews;
}

void CombinedBodyHudPage::onStateChange(CheckBox *box, bool newState)
{
	assert ( wireframeBoxes->size() == bodies->size() );

	for( size_t i=0; i<bodies->size(); i++ )
	{
		if ( (*wireframeBoxes)[i] == box )
		{
			(*bodies)[i]->setWireframeMode(newState);
			break;
		}
	}
}

void CombinedBodyHudPage::onStateChange(SelectView<Visualization> *view, Visualization t)
{
	assert ( visualizationTypeViews->size() == bodies->size() );

	for( size_t i=0; i<bodies->size(); i++ )
	{
		if ( (*visualizationTypeViews)[i] == view )
		{
			(*bodies)[i]->setVisualizationType(t);
			break;
		}
	}
}

void CombinedBodyHudPage::draw(DrawService *drawService)
{
	/* Updating infos */
	for( size_t i=0; i<bodies->size(); i++ )
	{
		Body *body = (*bodies)[i];

		/* Position */
		std::string position = *((*bodies)[i]->getName());
		position.append(" POSITION ");
		position.append(std::to_string(body->getCenter().x));
		position.append(", ");
		position.append(std::to_string(body->getCenter().y));
		position.append(", ");
		position.append(std::to_string(body->getCenter().z));

		(*positionViews)[i]->setText(position);

		/* Velocity */
		glm::dvec3 v = body->getVelocity();
		double vel = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
		std::string velocity = *((*bodies)[i]->getName());
		velocity.append(" VELOCITY ");
		velocity.append(std::to_string(vel));

		(*velocityViews)[i]->setText(velocity);
	}

	HudPage::draw(drawService);
}
