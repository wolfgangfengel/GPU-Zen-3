#ifndef _CAMERA_H_INCLUDED_
#define _CAMERA_H_INCLUDED_
//-------------------------------------------------------------------------------

#include <iostream>

#include "cyMatrix4.h"
#include "cyMatrix3.h"

//-------------------------------------------------------------------------------

#define DEG2RAD(v) ((v)*float(.017453292519943295f))

//-------------------------------------------------------------------------------

class CameraBase
{
private:
	cyMatrix4f matrix, matrixInverse;
	cyMatrix3f normal;
protected:
	cyMatrix4f proj;
	cyMatrix4f view;
public:

	const cyMatrix4f& GetMatrix() const { return matrix; }
	const cyMatrix4f& GetViewMatrix() const { return view; }
	const cyMatrix4f& GetProjMatrix() const { return proj; }
	const cyMatrix3f& GetNormalMatrix() const { return normal; }
	const cyMatrix4f& GetMatrixInverse() const { return matrixInverse; }

	void SetMatrix( const cyMatrix4f &viewMatrix, const cyMatrix4f &projMatrix ) 
	{ 
		view=viewMatrix; proj=projMatrix; 
	}
	void UpdateViewMatrix() { UpdateView(); UpdateMatrix(); }

private:
	void UpdateProjection() {
		ComputeProjectionMatrix();
	}
	void UpdateView() {
		ComputeViewMatrix();
		cyMatrix3f view3;
		view.GetSubMatrix3data( view3.data );
		normal = view3.GetInverse().GetTranspose();
		UpdateMatrix();
	}
	void UpdateMatrix() {
		matrix = proj * view;
		matrix.GetInverse(matrixInverse);
	}
protected:
	virtual void ComputeViewMatrix()=0;
	virtual void ComputeProjectionMatrix()=0;
	void UpdateProjectionMatrix() { UpdateProjection(); UpdateMatrix(); }
	
	void UpdateAllMatrices() { UpdateView(); UpdateProjection(); UpdateMatrix(); }
};


//-------------------------------------------------------------------------------

class ProjectionCamera : public CameraBase
{
protected:
	float fov;
	float aspect;
	float znear, zfar;
public:
	ProjectionCamera() : fov(40), aspect(1), znear(0.02f), zfar(150.0f) {}

	void SetAspect  ( float asp ) { aspect = asp; UpdateProjectionMatrix(); }

protected:
	void ComputeProjectionMatrix() { 
		proj.SetPerspective( DEG2RAD(fov), aspect, znear, zfar ); 
	}
};

//-------------------------------------------------------------------------------

class Camera : public ProjectionCamera
{
protected:
	float distance;
	
	cyPoint3f target;
	cyPoint2f offset;
public:
	cyPoint2f rot;
	Camera() : 
	distance(1.f), rot(3.14f * 0.35f, 3.14f * 0.15f), target(0.5, 0, 0.5), offset(0 ,0) 
	{ 
		UpdateAllMatrices(); 

		//cyPoint3f c = GetCameraLocation();
		//std::cout << "C " << c.x << " " << c.y << " " << c.z << std::endl;
	}

	void SetTarget(const cyPoint3f& p) { target = p; UpdateViewMatrix(); }
	void AddRotation(float x, float y) { rot.x += x; rot.y += y; UpdateViewMatrix(); }
	void AddDistance(float d) { distance += d; distance = fmaxf(distance, 0.1f); UpdateViewMatrix(); }
	void SetDistance(float d) { distance = d; distance = fmaxf(distance, 0.1f); UpdateViewMatrix(); }
	float GetDistance() const { return distance; }
	void AddOffset(float x, float y) { offset.x += x; offset.y += y; UpdateViewMatrix(); }

	cyPoint3f Get_Target() { return target; }
	cyPoint2f Get_Offset() { return offset; }
	cyPoint2f Get_Rotation() { return rot; }
	float Get_Distance() { return distance; }
	void SetRotation(float x, float y) { rot.x = x; rot.y = y; UpdateViewMatrix(); }
	void SetOffset(float x, float y) { offset.x = x; offset.y = y; UpdateViewMatrix(); }

	cyPoint3f GetCameraLocation() const
	{
		cyMatrix4f tview = cyMatrix4f::MatrixTrans(cyPoint3f(0, 0, -distance)) *
			cyMatrix4f::MatrixRotationX(rot.y) *
			cyMatrix4f::MatrixRotationY(rot.x) *
			cyMatrix4f::MatrixTrans(-target - cyPoint3f(offset.x, 0, offset.y));

		//ComputeViewMatrix();
		//
		//std::cout << view[0] << " " << view[1] << " " << view[2] << std::endl;
		//
		//cyPoint3f axisX  = cyPoint3f(tview[0], tview[4], tview[8]);
		//cyPoint3f axisY  = cyPoint3f(tview[1], tview[5], tview[9]);
		//cyPoint3f axisZ  = cyPoint3f(tview[2], tview[6], tview[10]);
		//cyPoint3f offset = cyPoint3f(tview[3], tview[7], tview[11]);
		//
		//cyPoint3f CameraLocation = cyPoint3f(-offset.Dot(axisX), -offset.Dot(axisY), -offset.Dot(axisZ));
		//
		//std::cout << CameraLocation.x << " " << CameraLocation.y << " " << CameraLocation.z << std::endl;

		cyPoint4f eye = cyPoint4f(0, 0, distance, 1) *
			cyMatrix4f::MatrixRotationX(rot.x) *
			cyMatrix4f::MatrixRotationY(rot.y) ;//*
			//cyMatrix4f::MatrixTrans(-target - cyPoint3f(offset.x, 0, offset.y));
		cyPoint3f CameraLocation = cyPoint3f(eye.x, eye.y, eye.z);
		return CameraLocation;
	}

protected:
	virtual void ComputeViewMatrix() {
		view = cyMatrix4f::MatrixTrans( cyPoint3f(0,0,-distance) ) * 
			   cyMatrix4f::MatrixRotationX(rot.y) * 
			   cyMatrix4f::MatrixRotationY(rot.x) * 
			   cyMatrix4f::MatrixTrans(-target-cyPoint3f(offset.x,0, offset.y));
	}
};

//-------------------------------------------------------------------------------

class DirLightCamera : public CameraBase
{
protected:
	float size;
	float znear, zfar;
	cyPoint3f target, direction;
public:
	DirLightCamera() : size(1), znear(-1), zfar(1) { proj.SetIdentity(); }

	void SetProjection(float width, float z_near, float z_far) { size = width; znear = z_near; zfar = z_far; UpdateProjectionMatrix(); }
	void SetTarget(const cyPoint3f& pos) { target = pos; UpdateViewMatrix(); }
	void SetDirection(const cyPoint3f& dir) { direction = dir; UpdateViewMatrix(); }

protected:
	void ComputeProjectionMatrix() {
		proj(0, 0) = 1.0f / size;
		proj(1, 1) = 1.0f / size;
		proj(2, 2) = 2.0f / (znear - zfar);
		proj(2, 3) = (znear + zfar) / (znear - zfar);
	}
	virtual void ComputeViewMatrix() {
		view.SetView(target, target - direction, cyPoint3f(0, 1, 0));
	}
};

//-------------------------------------------------------------------------------

#endif