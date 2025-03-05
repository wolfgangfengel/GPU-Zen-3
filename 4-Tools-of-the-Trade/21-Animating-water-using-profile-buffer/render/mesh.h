
#ifndef _MESH_
#define _MESH_

#include <vector>
#include <functional>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cu/helper_cuda.h"

#include "mathUtils.h"


struct Attributes {
	enum {
		Position = 0,
		Normal = 1,
		TexCoord = 2,
		Velocity = 3,
	};
};

struct Vertex {
	Vec3 position;
	Vec3 normal;
	Vec2 texCoord;
};

inline void setAttribPointer(GLuint vertexArrayObject, GLuint location, GLuint buffer, 
							 GLint size, GLenum type, GLboolean normalized, 
							 GLsizei stride, GLuint offset) 
{
	//assert((glIsBuffer(buffer) != GL_FALSE));

	GLint previousBuffer;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previousBuffer);
	{
		GLint previousVAO;
		glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVAO);
		{
			glBindVertexArray(vertexArrayObject);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glEnableVertexAttribArray(location);
			glVertexAttribPointer(location, size, type, normalized, stride, 
								  reinterpret_cast<void*>((long long)offset));
		}
		glBindVertexArray(previousVAO);
	}
	glBindBuffer(GL_ARRAY_BUFFER, previousBuffer);
}


inline std::vector<Vertex> createMeshVertices(unsigned width, unsigned length, float dx, 
											  std::function<float(Vec2)> heightFunction, bool setPos) 
{
	unsigned width1 = width + 1;
	unsigned length1 = length + 1;

	float d = 1.0f / width;

	std::vector<Vertex> vertices(width1 * length1);

	// Kui: no need to write data here since we will copy the data from gpu
	if (setPos)
	{

#pragma omp parallel for
		for (int x = 0; x < (int) width1; x++)
		{
			for (int y = 0; y < (int) length; y++)
			{
				auto normalizedPosition = Vec2(x * d, y * d);

				const auto NORMAL_EPSILON = 0.01f * d;
				auto epsilonPositionX = Vec2{ normalizedPosition.x, normalizedPosition.y } + Vec2{ NORMAL_EPSILON, 0 };
				auto epsilonPositionY = Vec2{ normalizedPosition.x, normalizedPosition.y } + Vec2{ 0, NORMAL_EPSILON };

				auto currentHeight = heightFunction(normalizedPosition / d);
				auto epsilonHeightX = heightFunction(epsilonPositionX / d);
				auto epsilonHeightY = heightFunction(epsilonPositionY / d);

				auto position = Vec3{ normalizedPosition.x, currentHeight, normalizedPosition.y };

				auto toEpsilonX = Vec3{ epsilonPositionX.x, epsilonHeightX, epsilonPositionX.y } - position;
				auto toEpsilonY = Vec3{ epsilonPositionY.x, epsilonHeightY, epsilonPositionY.y } - position;

				auto normal = toEpsilonY.Cross(toEpsilonX).GetNormalized();

				auto texCoord = Vec2(float(x), float(y)) / 50.0f;

				vertices[x * length1 + y] = (Vertex{ position, normal, texCoord });
			}
		}
	}
	return vertices;
}

inline std::vector<GLuint> createMeshIndices(unsigned width, unsigned length) {
	std::vector<GLuint> indices;
	unsigned verticesPerRow = width + 1;
	for0(x, width) 
	{
		for0(y, length) 
		{
#if 0
			// Triangle 1
			indices.push_back(x * verticesPerRow + y);
			indices.push_back(x * verticesPerRow + (y + 1));
			indices.push_back((x + 1) * verticesPerRow + (y + 1));

			// Triangle 2
			indices.push_back(x * verticesPerRow + y);
			indices.push_back((x + 1) * verticesPerRow + (y + 1));
			indices.push_back((x + 1) * verticesPerRow + y);
#else
			// Triangle 1
			indices.push_back(x + y * verticesPerRow);
			indices.push_back(x + (y + 1) * verticesPerRow);
			indices.push_back((x + 1) + (y + 1) * verticesPerRow);

			// Triangle 2
			indices.push_back(x + y * verticesPerRow);
			indices.push_back((x + 1) + (y + 1) * verticesPerRow);
			indices.push_back((x + 1) + y * verticesPerRow);
			
			
			
			
#endif
		}
	}
	return indices;
}

class Mesh {
public:

	Mesh() : 
		mWidth(0), 
		mLength(0), 
		mDx(1.0f), 
		mIndexCount(0),
		mPositionBuffer(-1),
		mNormalBuffer(-1),
		mVelocityBuffer(-1),
		mTexCoordBuffer(-1),
		mElementArrayBuffer(-1),
		mVertexArrayObject(-1)
	{}

	void genBuffers()
	{
		glGenBuffers(1, &mPositionBuffer);
		glGenBuffers(1, &mNormalBuffer);
		glGenBuffers(1, &mVelocityBuffer);
		glGenBuffers(1, &mTexCoordBuffer);
		glGenBuffers(1, &mElementArrayBuffer);

		glGenVertexArrays(1, &mVertexArrayObject);
		setAttribPointer(mVertexArrayObject, Attributes::Position, mPositionBuffer, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), 0);
		setAttribPointer(mVertexArrayObject, Attributes::Normal,   mNormalBuffer,   3, GL_FLOAT, GL_TRUE,  sizeof(Vec3), 0);
		setAttribPointer(mVertexArrayObject, Attributes::TexCoord, mTexCoordBuffer, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), 0);
		setAttribPointer(mVertexArrayObject, Attributes::Velocity, mVelocityBuffer, 3, GL_FLOAT, GL_TRUE,  sizeof(Vec4), 0);
	}

	void initialize(int width, int length, float dx, const std::function<float(Vec2)> heightFunction, bool setPos)
	{
		mWidth = width, mLength = length, mDx = dx;

		auto vertices = createMeshVertices(width, length, dx, heightFunction, setPos);
		auto indices = createMeshIndices(width, length);

		auto vertexCount = (width + 1) * (length + 1);
		mIndexCount = width * length * 6;

		std::vector<Vec3> positionData;
		for0(i, vertices.size()) 
		{
			positionData.push_back(vertices[i].position);
		}

		std::vector<Vec3> normalData;
		for0(i, vertices.size()) 
		{
			normalData.push_back(vertices[i].normal);
		}

		std::vector<Vec3> velocityData;
		for0(i, vertices.size()) 
		{
			velocityData.push_back(Vec3{ 0.0, 0.0, 0.0 });
		}

		std::vector<Vec2> texCoordData;
		for0(i, vertices.size()) 
		{
			texCoordData.push_back(vertices[i].texCoord);
		}
		glBindBuffer(GL_ARRAY_BUFFER, mPositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * vertexCount, positionData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * vertexCount, normalData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mVelocityBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3)* vertexCount, velocityData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mTexCoordBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec2) * vertexCount, texCoordData.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementArrayBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * mIndexCount, indices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPosResource, mPositionBuffer, cudaGraphicsMapFlagsNone));
		getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaNormResource, mNormalBuffer, cudaGraphicsMapFlagsNone));
		getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	}

	void initialize(int width, int length, float dx, std::vector<GLuint>& indices, std::vector<Vertex>& vertices)
	{
		mWidth = width, mLength = length, mDx = dx;
		//auto vertices = createMeshVertices(width, length, dx, heightFunction, setPos);
		//auto indices = createMeshIndices(width, length);

		//auto vertexCount = (width + 1) * (length + 1);
		mIndexCount = width * length * 6;
		int vertexCount = mIndexCount;

		std::vector<Vec3> positionData;
		for0(i, vertices.size())
		{
			positionData.push_back(vertices[i].position);
		}

		std::vector<Vec3> normalData;
		for0(i, vertices.size())
		{
			normalData.push_back(vertices[i].normal);
		}

		std::vector<Vec3> velocityData;
		for0(i, vertices.size())
		{
			velocityData.push_back(Vec3{ 0.0, 0.0, 0.0 });
		}

		std::vector<Vec2> texCoordData;
		for0(i, vertices.size())
		{
			texCoordData.push_back(vertices[i].texCoord);
		}

		glBindBuffer(GL_ARRAY_BUFFER, mPositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * vertexCount, positionData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * vertexCount, normalData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mVelocityBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * vertexCount, velocityData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, mTexCoordBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec2) * vertexCount, texCoordData.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementArrayBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * mIndexCount, indices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPosResource, mPositionBuffer, cudaGraphicsMapFlagsNone));
		getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaNormResource, mNormalBuffer, cudaGraphicsMapFlagsNone));
		getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
	}


	void UpdatePosition(const std::function<float(Vec2)> heightFunction)
	{
		auto vertices = createMeshVertices(mWidth, mLength, mDx, heightFunction, true);
	
		auto vertexCount = (mWidth + 1) * (mLength + 1);
	
		std::vector<Vec3> positionData;
		for0(i, vertices.size()) {
			positionData.push_back(vertices[i].position);
		}

		std::vector<Vec3> normalData;
		for0(i, vertices.size()) {
			normalData.push_back(vertices[i].normal);
		}

		glBindBuffer(GL_ARRAY_BUFFER, mPositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec4) * vertexCount, positionData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vec4)* vertexCount, normalData.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void render() const 
	{
		//glDisable(GL_CULL_FACE);
		glBindVertexArray(mVertexArrayObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementArrayBuffer);
		glDrawElements(GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, nullptr);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	GLuint getPositionBuffer() const { return mPositionBuffer; }
	GLuint getNormalBuffer() const { return mNormalBuffer; }

	cudaGraphicsResource* getCudaPos() { return cudaPosResource; }
	cudaGraphicsResource* getCudaNorm() { return cudaNormResource; }

private:
	int			mWidth, mLength;
	float		mDx;
	GLuint		mPositionBuffer;
	GLuint		mNormalBuffer;
	GLuint		mVelocityBuffer;
	GLuint		mTexCoordBuffer;
	GLuint		mElementArrayBuffer;
	GLuint		mVertexArrayObject;
	unsigned	mIndexCount;

	struct cudaGraphicsResource* cudaPosResource;
	struct cudaGraphicsResource* cudaNormResource;
};

#endif