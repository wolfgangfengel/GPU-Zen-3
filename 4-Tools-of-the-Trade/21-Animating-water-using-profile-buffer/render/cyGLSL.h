// cyCodeBase by Cem Yuksel
// [www.cemyuksel.com]
//-------------------------------------------------------------------------------
///
/// \file		cyGLSL.h 
/// \author		Cem Yuksel
/// \version	1.0
/// \date		October 25, 2015
///
/// \brief Helper classes for GLSL shaders
///
//-------------------------------------------------------------------------------

#ifndef _CY_GLSL_H_INCLUDED_
#define _CY_GLSL_H_INCLUDED_

//-------------------------------------------------------------------------------

#include <GL/glew.h>
#include <GL/GL.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//-------------------------------------------------------------------------------

#define CY_GLSL_INVALID_ID 0xFFFFFFFF

//-------------------------------------------------------------------------------

/// Shader class
class cyGLSLShader
{
private:
	GLuint shaderID;
public:
	cyGLSLShader() : shaderID(CY_GLSL_INVALID_ID) {}
	virtual ~cyGLSLShader() { Clear(); }

	GLuint ShaderID() const { return shaderID; }

	void Clear()
	{
		if ( shaderID != CY_GLSL_INVALID_ID ) {
			glDeleteShader(shaderID);
			shaderID = CY_GLSL_INVALID_ID;
		}
	}

	bool CompileShaderFile( const char *filename, GLenum shaderType, std::ostream &outStream=std::cout )
	{
		std::ifstream shaderStream(filename, std::ios::in);
		if(! shaderStream.is_open()) {
			outStream << "ERROR: Cannot open file." << std::endl;
			return false;
		}

		std::string shaderSourceCode((std::istreambuf_iterator<char>(shaderStream)), std::istreambuf_iterator<char>());
		shaderStream.close();

		return CompileShaderSource( shaderSourceCode.data(), shaderType, outStream );
	}

	bool CompileShaderSource( const char *shaderSourceCode, GLenum shaderType, std::ostream &outStream=std::cout )
	{
		Clear();

		shaderID = glCreateShader( shaderType );
		glShaderSource(shaderID, 1, &shaderSourceCode, NULL);
		glCompileShader(shaderID);

		GLint result = GL_FALSE;
		glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);

		int infoLogLength;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
		if ( infoLogLength > 1 ) {
			std::vector<char> compilerMessage(infoLogLength);
			glGetShaderInfoLog( shaderID, infoLogLength, NULL, compilerMessage.data() );
			outStream << "ERROR: " << compilerMessage.data() << std::endl;
		}

		if ( result ) {
			GLint stype;
			glGetShaderiv(shaderID, GL_SHADER_TYPE, &stype);
			if ( stype != shaderType ) {
				outStream << "ERROR: Incorrect shader type." << std::endl;
				return false;
			}
		}

		return result == GL_TRUE;
	}
};

//-------------------------------------------------------------------------------

class cyGLSLProgram
{
private:
	GLuint programID;
	std::vector<GLint> params;

public:
	cyGLSLProgram() : programID(CY_GLSL_INVALID_ID) {}
	virtual ~cyGLSLProgram() { Clear(); }

	GLuint ProgramID() const { return programID; }

	void Clear()
	{
		if ( programID != CY_GLSL_INVALID_ID ) {
			glDeleteProgram(programID);
			programID = CY_GLSL_INVALID_ID;
		}
	}

	void CreateProgram() { Clear(); programID = glCreateProgram(); }

	void BindProgram() { glUseProgram(programID); }

	void AttachShader( const cyGLSLShader &shader ) { AttachShader(shader.ShaderID()); }
	void AttachShader( GLuint shaderID ) { glAttachShader(programID,shaderID); }

	bool LinkProgram( std::ostream &outStream=std::cout )
	{
		glLinkProgram(programID);

		GLint result = GL_FALSE;
		glGetProgramiv(programID, GL_LINK_STATUS, &result);

		int infoLogLength;
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
		if ( infoLogLength > 1 ) {
			std::vector<char> compilerMessage(infoLogLength);
			glGetProgramInfoLog( programID, infoLogLength, NULL, compilerMessage.data() );
			outStream << "ERROR: " << compilerMessage.data() << std::endl;
		}

		return result == GL_TRUE;
	}

	bool BuildProgramFiles( const char *vertexShaderFile, 
                            const char *fragmentShaderFile,
							const char *geometryShaderFile=NULL,
							const char *tessControlShaderFile=NULL,
							const char *tessEvaluationShaderFile=NULL,
							std::ostream &outStream=std::cout )
	{
		Clear();
		CreateProgram();
		cyGLSLShader vs, fs, gs, tcs, tes;
		std::stringstream output;
		if ( ! vs.CompileShaderFile(vertexShaderFile, GL_VERTEX_SHADER, output) ) {
			outStream << "ERROR: Failed compiling vertex shader \"" << vertexShaderFile << ".\"" << std::endl << output.str();
			return false;
		}
		AttachShader(vs);
		if ( ! fs.CompileShaderFile(fragmentShaderFile, GL_FRAGMENT_SHADER, output) ) {
			outStream << "ERROR: Failed compiling fragment shader \"" << fragmentShaderFile << ".\"" <<  std::endl << output.str();
			return false;
		}
		AttachShader(fs);
		if ( geometryShaderFile ) {
			if ( ! gs.CompileShaderFile(geometryShaderFile, GL_GEOMETRY_SHADER, output) ) {
				outStream << "ERROR: Failed compiling geometry shader \"" << geometryShaderFile << ".\"" <<  std::endl << output.str();
				return false;
			}
			AttachShader(gs);
		}
		if ( tessControlShaderFile ) {
			if ( ! tcs.CompileShaderFile(tessControlShaderFile, GL_TESS_CONTROL_SHADER, output) ) {
				outStream << "ERROR: Failed compiling tessellation control shader \"" << tessControlShaderFile << ".\"" <<  std::endl << output.str();
				return false;
			}
			AttachShader(tcs);
		}
		if ( tessEvaluationShaderFile ) {
			if ( ! tes.CompileShaderFile(tessEvaluationShaderFile, GL_TESS_EVALUATION_SHADER, output) ) {
				outStream << "ERROR: Failed compiling tessellation evaluation shader \"" << tessEvaluationShaderFile << ".\"" <<  std::endl << output.str();
				return false;
			}
			AttachShader(tes);
		}
		LinkProgram(outStream);
		return true;
	}

	bool BuildProgramSources( const char *vertexShaderSourceCode, 
                              const char *fragmentShaderSourceCode,
							  const char *geometryShaderSourceCode=NULL,
							  const char *tessControlShaderSourceCode=NULL,
							  const char *tessEvaluationShaderSourceCode=NULL,
							  std::ostream &outStream=std::cout )
	{
		Clear();
		CreateProgram();
		cyGLSLShader vs, fs, gs, tcs, tes;
		std::stringstream output;
		if ( ! vs.CompileShaderSource(vertexShaderSourceCode, GL_VERTEX_SHADER, output) ) {
			outStream << "ERROR: Failed compiling vertex shader." << std::endl << output.str();
			return false;
		}
		AttachShader(vs);
		if ( ! fs.CompileShaderSource(fragmentShaderSourceCode, GL_FRAGMENT_SHADER, output) ) {
			outStream << "ERROR: Failed compiling fragment shader." << std::endl << output.str();
			return false;
		}
		AttachShader(fs);
		if ( geometryShaderSourceCode ) {
			if ( ! gs.CompileShaderSource(geometryShaderSourceCode, GL_GEOMETRY_SHADER, output) ) {
				outStream << "ERROR: Failed compiling geometry shader." << std::endl << output.str();
				return false;
			}
			AttachShader(gs);
		}
		if ( tessControlShaderSourceCode ) {
			if ( ! tcs.CompileShaderSource(tessControlShaderSourceCode, GL_TESS_CONTROL_SHADER, output) ) {
				outStream << "ERROR: Failed compiling tessellation control shader." << std::endl << output.str();
				return false;
			}
			AttachShader(tcs);
		}
		if ( tessEvaluationShaderSourceCode ) {
			if ( ! tes.CompileShaderSource(tessEvaluationShaderSourceCode, GL_TESS_EVALUATION_SHADER, output) ) {
				outStream << "ERROR: Failed compiling tessellation evaluation shader." << std::endl << output.str();
				return false;
			}
			AttachShader(tes);
		}
		LinkProgram(outStream);
		return true;
	}

	/// Registers a single parameter.
	/// The id must be unique and the name should match a uniform parameter name in one of the shaders.
	/// The id values for different parameters don't have to be consecutive, but unused id values waste memory.
	void RegisterParam( unsigned int id, const char *name, std::ostream &outStream=std::cout )
	{
		if ( params.size() <= id ) params.resize( (int)id+1, -1 );
		params[id] = glGetUniformLocation( programID, name );
		if ( params[id] < 0 ) {
			GLenum error = glGetError();
			GLenum newError;
			while ( (newError = glGetError()) != GL_NO_ERROR ) error = newError; // get the latest error.
			outStream << "OpenGL ERROR: " << gluErrorString(error) << ". Parameter \"" << name << "\" could not be registered." << std::endl;
		}
	}

	/// Registers multiple parameters.
	/// The names should be separated by a space character.
	void RegisterParams( const char *names, unsigned int startingID=0, std::ostream &outStream=std::cout )
	{
		std::stringstream ss(names);
		unsigned int id = startingID;
		while ( ss.good() ) {
			std::string name;
			ss >> name;
			RegisterParam( id++, name.c_str(), outStream );
		}
	}

	void SetParam(unsigned int paramID, float x)							{ glUniform1f(params[paramID],x); }
	void SetParam(unsigned int paramID, float x, float y)					{ glUniform2f(params[paramID],x,y); }
	void SetParam(unsigned int paramID, float x, float y, float z)			{ glUniform3f(params[paramID],x,y,z); }
	void SetParam(unsigned int paramID, float x, float y, float z, float w)	{ glUniform4f(params[paramID],x,y,z,w); }
	void SetParam(unsigned int paramID, int x)								{ glUniform1i(params[paramID],x); }
	void SetParam(unsigned int paramID, int x, int y)						{ glUniform2i(params[paramID],x,y); }
	void SetParam(unsigned int paramID, int x, int y, int z)				{ glUniform3i(params[paramID],x,y,z); }
	void SetParam(unsigned int paramID, int x, int y, int z, int w)			{ glUniform4i(params[paramID],x,y,z,w); }
	void SetParam(unsigned int paramID, unsigned int x)													{ glUniform1ui(params[paramID],x); }
	void SetParam(unsigned int paramID, unsigned int x, unsigned int y)									{ glUniform2ui(params[paramID],x,y); }
	void SetParam(unsigned int paramID, unsigned int x, unsigned int y, unsigned int z)					{ glUniform3ui(params[paramID],x,y,z); }
	void SetParam(unsigned int paramID, unsigned int x, unsigned int y, unsigned int z, unsigned int w)	{ glUniform4ui(params[paramID],x,y,z,w); }
	void SetParamMatrix2  (unsigned int paramID, const float *m) { glUniformMatrix2fv  (params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix2x3(unsigned int paramID, const float *m) { glUniformMatrix2x3fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix2x4(unsigned int paramID, const float *m) { glUniformMatrix2x4fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix3x2(unsigned int paramID, const float *m) { glUniformMatrix3x2fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix3  (unsigned int paramID, const float *m) { glUniformMatrix3fv  (params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix3x4(unsigned int paramID, const float *m) { glUniformMatrix3x4fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix4x2(unsigned int paramID, const float *m) { glUniformMatrix4x2fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix4x3(unsigned int paramID, const float *m) { glUniformMatrix4x3fv(params[paramID],1,GL_FALSE,m); }
	void SetParamMatrix4  (unsigned int paramID, const float *m) { glUniformMatrix4fv  (params[paramID],1,GL_FALSE,m); }

};

//-------------------------------------------------------------------------------
#endif