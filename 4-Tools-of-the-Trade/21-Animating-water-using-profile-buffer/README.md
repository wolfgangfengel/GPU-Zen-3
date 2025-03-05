# Animating Water using Profile Buffer
This is the official implementation for the paper *Animating Water using Profile Buffer*, presented by Haozhe Su, Wei Li, Zherong Pan, Xifeng Gao, Zhenyu Mao, Kui Wu, from Tencent America, LightSpeed Studios.

## Build
```
mkdir build
cd build
cmake-gui.exe ..
```
Then we can configure and generate VS project: 
1. Make sure the source path and the build path are both set up correctly:
![](./install/cmakePaths.png) 
2. Click on the **Configure** button to proceed:
![](./install/cmakeConfigure.png) 
3. If everything works just fine, click on the **Generate** button to generate a VS project:
![](./install/cmakeGenerate.png)
4. Open the generated VS project by clicking on **Open Project**:
![](./install/cmakeOpenProject.png)
5. Set **CuSSWE** as startup project:
![](./install/vsSetStartup.png)
6. Switch to **Release** mode and then build solution:
![](./install/vsRelease.png)
![](./install/vsBuild.png)

## Run
1. Set up command line arguments as **River**:
![](./install/vsArgument.png)
![](./install/vsArguments.png)
2. Copy the following DLLs into **Release** folder:
![](./install/sourceDLLs.png)
![](./install/destDLLs.png)
3. Run the project:
![](./install/vsRun.png)
If everything has been set up properly, the following scene should pop up and we can press **Space** to start/pause the animation.
![](./install/river.png)
