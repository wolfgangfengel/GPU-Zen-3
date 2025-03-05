#ifndef _SHADOW_H_INCLUDED_
#define _SHADOW_H_INCLUDED_
//-------------------------------------------------------------------------------

#include "camera.h"
#include <GL/glew.h>
#include <GL/GL.h>

//-------------------------------------------------------------------------------

class DirShadow
{
private:
	DirLightCamera camera;
	GLuint m_fbo = -1;
	GLuint m_shadowMap = -1;
	GLuint width = -1, height = -1;
	cyMatrix4f lookupMatrix;

public:

	bool Init(int resolution, const cyPoint3f &lightDir, float size, bool depthLookup=true, GLenum format=GL_DEPTH_COMPONENT)
	{
		camera.SetProjection( size, -size, size );
		camera.SetDirection( lightDir );
		UpdateLookupMatrix();

		width = resolution;
		height = resolution;

		glGenFramebuffers(1, &m_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

		glGenTextures(1, &m_shadowMap);
		glBindTexture(GL_TEXTURE_2D, m_shadowMap);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		if ( depthLookup ) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
		} else {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}
		float border[]={1,0,0,0};
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

		if ( format == GL_DEPTH_COMPONENT ) {
			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_shadowMap, 0);
			glDrawBuffer(GL_NONE);
		} else {
			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_shadowMap, 0);
			/*
			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
			glEnable(GL_DEPTH_TEST);
			*/
			GLuint depthrenderbuffer;
			glGenRenderbuffers(1, &depthrenderbuffer);
			glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

			GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
			glDrawBuffers(1, drawBuffers);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		if( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ) return false;
		return true;
	}

	void BeginRenderShadow()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		glViewport(0,0,width,height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void EndRenderShadow()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void SetTarget( const cyPoint3f &target ) { camera.SetTarget(target); UpdateLookupMatrix(); }

	const cyMatrix4f& GetMatrix() const { return camera.GetMatrix(); }
	const cyMatrix4f& GetViewMatrix() const { return camera.GetViewMatrix(); }
	const cyMatrix4f& GetProjMatrix() const { return camera.GetProjMatrix(); }

	const cyMatrix4f& GetLookupMatrix() const { return lookupMatrix; }

	void BindTexture() { glBindTexture(GL_TEXTURE_2D, m_shadowMap); }

private:
	void UpdateLookupMatrix()
	{
		cyMatrix4f lookup( cyPoint4f(0.5f,0,0,0), cyPoint4f(0,0.5f,0,0), cyPoint4f(0,0,0.5f,0), cyPoint4f(0.5f,0.5f,0.5f,1) );
		lookupMatrix = lookup * camera.GetMatrix();
	}
};

//-------------------------------------------------------------------------------
#endif