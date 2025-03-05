#pragma once

#include <fstream>

#include "scene/scene.h"

class TestSwirlScene : public Scene {
public:
    TestSwirlScene() : Scene("Swirl Scene") {}

    void Initialize();

    void Initialize_Water_Depth_Field();
    void Initialize_Solid_Field();

private:

};

