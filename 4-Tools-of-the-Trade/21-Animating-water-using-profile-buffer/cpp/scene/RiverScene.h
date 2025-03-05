#pragma once

#include <fstream>

#include "scene/scene.h"

class RiverScene : public Scene {
public:
    RiverScene() : Scene("River Scene") {}

    void Initialize();

    void Initialize_Water_Depth_Field();
    void Initialize_Solid_Field();

private:

};

