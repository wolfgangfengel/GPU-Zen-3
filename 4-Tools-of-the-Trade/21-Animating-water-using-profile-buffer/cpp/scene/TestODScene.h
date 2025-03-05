#pragma once

#include <fstream>

#include "scene/scene.h"

class TestODScene : public Scene {
public:
    TestODScene() : Scene("Opposite Direction Scene") {}

    void Initialize();

    void Initialize_Water_Depth_Field();
    void Initialize_Solid_Field();

private:

};

