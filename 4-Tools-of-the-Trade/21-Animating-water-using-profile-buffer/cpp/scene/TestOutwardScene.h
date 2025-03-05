#pragma once

#include <fstream>

#include "scene/scene.h"

/** \brief Test scene that the water flow outwards.
*/
class TestOutwardScene : public Scene {
public:
    TestOutwardScene() : Scene("Outward Scene") {}

    void Initialize();

    void Initialize_Water_Depth_Field();
    void Initialize_Solid_Field();

private:

};

