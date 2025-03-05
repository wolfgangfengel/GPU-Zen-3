#pragma once

#include <fstream>

#include "scene/scene.h"

class TestFDScene : public Scene {
public:
    TestFDScene() : Scene("Fixed Direction Scene") {}

    void Initialize();

    void Initialize_Water_Depth_Field();
    void Initialize_Solid_Field();

private:

};

